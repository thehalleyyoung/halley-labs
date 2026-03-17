// regsynth-solver: DPLL/CDCL SAT solver
// Full CDCL implementation with watched literals, VSIDS, 1UIP conflict analysis,
// non-chronological backtracking, clause learning, and UNSAT core extraction.

use crate::result::{Assignment, Clause, Literal, SatResult, SolverStatistics, Variable, lit_neg, lit_sign, lit_var, make_lit};
use crate::solver_config::SolverConfig;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

// ─── Clause Database ────────────────────────────────────────────────────────

/// Internal clause representation.
#[derive(Debug, Clone)]
pub struct ClauseInfo {
    pub lits: Vec<Literal>,
    pub learned: bool,
    pub activity: f64,
    /// Index of the original input clause (for UNSAT core tracking).
    /// Learned clauses store the set of original clause indices they derive from.
    pub provenance: HashSet<usize>,
}

// ─── Trail Entry ────────────────────────────────────────────────────────────

/// An entry on the assignment trail.
#[derive(Debug, Clone, Copy)]
struct TrailEntry {
    lit: Literal,
    decision_level: u32,
    /// Clause index that implied this literal, or `usize::MAX` for decisions.
    reason: usize,
}

const NO_REASON: usize = usize::MAX;

// ─── Watched Literals ───────────────────────────────────────────────────────

/// Two-literal watching scheme: each clause is watched by exactly two literals.
#[derive(Debug, Clone, Copy)]
struct WatcherInfo {
    clause_idx: usize,
    blocker: Literal,
}

// ─── Value ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LBool {
    True,
    False,
    Undef,
}

// ─── DPLL Solver ────────────────────────────────────────────────────────────

/// CDCL SAT solver with watched literals, VSIDS, and 1UIP conflict analysis.
pub struct DpllSolver {
    num_vars: u32,
    clauses: Vec<ClauseInfo>,
    /// Assignment for each variable (1-indexed): True, False, or Undef.
    assigns: Vec<LBool>,
    /// Decision level for each variable.
    var_level: Vec<u32>,
    /// Reason clause for each variable (NO_REASON for decisions/unassigned).
    var_reason: Vec<usize>,
    /// The assignment trail (in order of assignment).
    trail: Vec<TrailEntry>,
    /// trail_lim[dl] = first index in trail at decision level dl.
    trail_lim: Vec<usize>,
    /// Current decision level.
    decision_level: u32,
    /// Watched literals: for each literal (indexed by lit_to_idx), list of watchers.
    watchers: Vec<Vec<WatcherInfo>>,
    /// Propagation queue (indices into trail).
    prop_queue: VecDeque<usize>,
    /// VSIDS activity scores.
    activity: Vec<f64>,
    /// VSIDS increment.
    var_inc: f64,
    /// Phase saving: last polarity assigned to each variable.
    saved_phase: Vec<bool>,
    /// Solver configuration.
    config: SolverConfig,
    /// Statistics.
    pub stats: SolverStatistics,
    /// Number of original (non-learned) clauses.
    num_original: usize,
    /// Root-level conflict detected during clause addition.
    root_conflict: Option<usize>,
}

impl DpllSolver {
    /// Create a new solver for `num_vars` variables.
    pub fn new(num_vars: u32, config: SolverConfig) -> Self {
        let n = num_vars as usize + 1; // 1-indexed
        let num_lits = 2 * n;
        Self {
            num_vars,
            clauses: Vec::new(),
            assigns: vec![LBool::Undef; n],
            var_level: vec![0; n],
            var_reason: vec![NO_REASON; n],
            trail: Vec::new(),
            trail_lim: Vec::new(),
            decision_level: 0,
            watchers: vec![Vec::new(); num_lits],
            prop_queue: VecDeque::new(),
            activity: vec![0.0; n],
            var_inc: 1.0,
            saved_phase: vec![false; n],
            config,
            stats: SolverStatistics::new(),
            num_original: 0,
            root_conflict: None,
        }
    }

    /// Index mapping: literal -> watcher list index.
    #[inline]
    fn lit_to_idx(lit: Literal) -> usize {
        if lit > 0 {
            2 * lit as usize
        } else {
            2 * (-lit) as usize + 1
        }
    }

    /// Get the current value of a literal under the assignment.
    #[inline]
    fn lit_value(&self, lit: Literal) -> LBool {
        let v = lit_var(lit) as usize;
        match self.assigns[v] {
            LBool::Undef => LBool::Undef,
            LBool::True => {
                if lit_sign(lit) {
                    LBool::True
                } else {
                    LBool::False
                }
            }
            LBool::False => {
                if lit_sign(lit) {
                    LBool::False
                } else {
                    LBool::True
                }
            }
        }
    }

    /// Add a clause to the solver. Returns the clause index.
    pub fn add_clause(&mut self, lits: Vec<Literal>) -> usize {
        self.add_clause_internal(lits, false, None)
    }

    /// Add an original clause with a provenance index.
    pub fn add_original_clause(&mut self, lits: Vec<Literal>, original_idx: usize) -> usize {
        let mut prov = HashSet::new();
        prov.insert(original_idx);
        self.add_clause_internal(lits, false, Some(prov))
    }

    fn add_clause_internal(
        &mut self,
        mut lits: Vec<Literal>,
        learned: bool,
        provenance: Option<HashSet<usize>>,
    ) -> usize {
        // Remove duplicates and trivially true clauses
        lits.sort_by_key(|l| (lit_var(*l), !lit_sign(*l)));
        lits.dedup();
        for i in 0..lits.len() {
            for j in (i + 1)..lits.len() {
                if lits[i] == lit_neg(lits[j]) {
                    // Tautological clause
                    let idx = self.clauses.len();
                    self.clauses.push(ClauseInfo {
                        lits,
                        learned,
                        activity: 0.0,
                        provenance: provenance.unwrap_or_default(),
                    });
                    return idx;
                }
            }
        }

        let idx = self.clauses.len();
        let prov = provenance.unwrap_or_default();

        if lits.is_empty() {
            self.clauses.push(ClauseInfo {
                lits,
                learned,
                activity: 0.0,
                provenance: prov,
            });
            return idx;
        }

        if lits.len() == 1 {
            // Unit clause: enqueue immediately or detect conflict
            self.clauses.push(ClauseInfo {
                lits: lits.clone(),
                learned,
                activity: 0.0,
                provenance: prov,
            });
            match self.lit_value(lits[0]) {
                LBool::Undef => self.enqueue(lits[0], idx),
                LBool::False => {
                    // Conflict: unit clause contradicts current assignment
                    self.root_conflict = Some(idx);
                }
                LBool::True => {} // Already satisfied
            }
            return idx;
        }

        // Set up watched literals: watch the first two literals
        let w0 = WatcherInfo {
            clause_idx: idx,
            blocker: lits[1],
        };
        let w1 = WatcherInfo {
            clause_idx: idx,
            blocker: lits[0],
        };
        self.watchers[Self::lit_to_idx(lit_neg(lits[0]))].push(w0);
        self.watchers[Self::lit_to_idx(lit_neg(lits[1]))].push(w1);

        self.clauses.push(ClauseInfo {
            lits,
            learned,
            activity: 0.0,
            provenance: prov,
        });
        idx
    }

    /// Enqueue a literal assignment with a reason clause.
    fn enqueue(&mut self, lit: Literal, reason: usize) {
        let var = lit_var(lit) as usize;
        self.assigns[var] = if lit_sign(lit) {
            LBool::True
        } else {
            LBool::False
        };
        self.var_level[var] = self.decision_level;
        self.var_reason[var] = reason;
        let entry = TrailEntry {
            lit,
            decision_level: self.decision_level,
            reason,
        };
        let trail_idx = self.trail.len();
        self.trail.push(entry);
        self.prop_queue.push_back(trail_idx);
    }

    /// Perform Boolean Constraint Propagation. Returns conflicting clause index or None.
    pub fn propagate(&mut self) -> Option<usize> {
        while let Some(trail_idx) = self.prop_queue.pop_front() {
            let p = self.trail[trail_idx].lit;
            self.stats.propagations += 1;

            let watch_idx = Self::lit_to_idx(p);
            let mut watchers = std::mem::take(&mut self.watchers[watch_idx]);
            let mut i = 0;
            let mut conflict = None;

            while i < watchers.len() {
                let watcher = watchers[i];
                let ci = watcher.clause_idx;

                // Check blocker first
                if self.lit_value(watcher.blocker) == LBool::True {
                    i += 1;
                    continue;
                }

                let clause_lits = self.clauses[ci].lits.clone();
                if clause_lits.len() < 2 {
                    i += 1;
                    continue;
                }

                // Make sure the false literal is in position 1
                let mut lit0 = clause_lits[0];
                let mut lit1 = clause_lits[1];
                if lit0 == lit_neg(p) {
                    // Swap positions 0 and 1
                    self.clauses[ci].lits[0] = lit1;
                    self.clauses[ci].lits[1] = lit0;
                    std::mem::swap(&mut lit0, &mut lit1);
                }

                // If the first watch is already true, clause is satisfied
                if self.lit_value(lit0) == LBool::True {
                    watchers[i].blocker = lit0;
                    i += 1;
                    continue;
                }

                // Look for a new literal to watch
                let mut found_new_watch = false;
                let clen = self.clauses[ci].lits.len();
                for k in 2..clen {
                    let litk = self.clauses[ci].lits[k];
                    if self.lit_value(litk) != LBool::False {
                        // Swap lits[1] and lits[k]
                        self.clauses[ci].lits[1] = litk;
                        self.clauses[ci].lits[k] = lit1;
                        // Add watcher for the new watched literal
                        let new_watcher = WatcherInfo {
                            clause_idx: ci,
                            blocker: lit0,
                        };
                        self.watchers[Self::lit_to_idx(lit_neg(litk))].push(new_watcher);
                        // Remove this watcher
                        watchers.swap_remove(i);
                        found_new_watch = true;
                        break;
                    }
                }

                if found_new_watch {
                    continue;
                }

                // No new watch found: clause is unit or conflicting
                if self.lit_value(lit0) == LBool::False {
                    // Conflict! Put remaining watchers back
                    conflict = Some(ci);
                    break;
                } else {
                    // Unit propagation
                    watchers[i].blocker = lit0;
                    self.enqueue(lit0, ci);
                    i += 1;
                }
            }

            // Put watchers back
            self.watchers[watch_idx] = watchers;

            if conflict.is_some() {
                self.prop_queue.clear();
                return conflict;
            }
        }
        None
    }

    /// Make a decision: pick an unassigned variable using VSIDS.
    pub fn decide(&mut self) -> bool {
        let mut best_var: Option<u32> = None;
        let mut best_act = -1.0f64;

        for v in 1..=self.num_vars {
            if self.assigns[v as usize] == LBool::Undef {
                let act = self.activity[v as usize];
                if act > best_act {
                    best_act = act;
                    best_var = Some(v);
                }
            }
        }

        let var = match best_var {
            Some(v) => v,
            None => return false, // All variables assigned
        };

        self.stats.decisions += 1;
        self.decision_level += 1;
        self.trail_lim.push(self.trail.len());

        let polarity = if self.config.phase_saving {
            self.saved_phase[var as usize]
        } else {
            false
        };
        let lit = make_lit(var, polarity);
        self.enqueue(lit, NO_REASON);
        true
    }

    /// Analyze a conflict clause (1UIP): returns (learned_clause, backtrack_level).
    pub fn analyze_conflict(&mut self, conflict_clause: usize) -> (Vec<Literal>, u32) {
        let mut learned = Vec::new();
        let mut seen = vec![false; self.num_vars as usize + 1];
        let mut counter = 0u32; // how many literals at the current decision level remain to resolve
        let mut p: Option<Literal> = None;
        let mut reason_clause = conflict_clause;
        let mut combined_provenance = HashSet::new();

        // Walk the trail backwards
        let mut trail_pos = self.trail.len();

        loop {
            // Gather provenance from reason clause
            combined_provenance.extend(self.clauses[reason_clause].provenance.iter().copied());

            // Bump activity for all variables in the reason clause
            let reason_lits: Vec<Literal> = self.clauses[reason_clause].lits.clone();
            for &lit in &reason_lits {
                let v = lit_var(lit) as usize;
                if !seen[v] {
                    seen[v] = true;
                    self.bump_var_activity(v);
                    if self.var_level[v] == self.decision_level {
                        counter += 1;
                    } else if self.var_level[v] > 0 {
                        learned.push(lit_neg(lit));
                    }
                }
            }

            // Find the next literal on the trail at the current decision level
            loop {
                trail_pos -= 1;
                let entry = self.trail[trail_pos];
                let v = lit_var(entry.lit) as usize;
                if seen[v] {
                    p = Some(entry.lit);
                    break;
                }
            }

            counter -= 1;
            if counter == 0 {
                break;
            }

            let pvar = lit_var(p.unwrap()) as usize;
            reason_clause = self.var_reason[pvar];
        }

        // The 1UIP literal is the negation of p
        learned.insert(0, lit_neg(p.unwrap()));

        // Compute backtrack level: second-highest decision level in learned clause
        let bt_level = if learned.len() == 1 {
            0
        } else {
            let mut max_level = 0u32;
            let mut max_idx = 1;
            for i in 1..learned.len() {
                let lv = self.var_level[lit_var(learned[i]) as usize];
                if lv > max_level {
                    max_level = lv;
                    max_idx = i;
                }
            }
            // Move the literal with the highest level to position 1 for watched literals
            learned.swap(1, max_idx);
            max_level
        };

        // Record provenance on the learned clause
        let ci = self.add_clause_internal(learned.clone(), true, Some(combined_provenance));
        self.clauses[ci].activity = 1.0;
        self.stats.learned_clauses += 1;

        (learned, bt_level)
    }

    /// Backtrack to the given decision level, undoing all assignments above it.
    pub fn backtrack(&mut self, level: u32) {
        if self.decision_level <= level {
            return;
        }
        let target_trail_len = if level == 0 {
            if self.trail_lim.is_empty() {
                self.trail.len()
            } else {
                self.trail_lim[0]
            }
        } else {
            self.trail_lim[level as usize]
        };

        // Undo assignments from the trail
        for i in (target_trail_len..self.trail.len()).rev() {
            let entry = self.trail[i];
            let var = lit_var(entry.lit) as usize;
            self.saved_phase[var] = lit_sign(entry.lit);
            self.assigns[var] = LBool::Undef;
            self.var_reason[var] = NO_REASON;
        }
        self.trail.truncate(target_trail_len);
        self.trail_lim.truncate(level as usize);
        self.decision_level = level;
        self.prop_queue.clear();
    }

    /// Bump a variable's VSIDS activity.
    fn bump_var_activity(&mut self, var: usize) {
        self.activity[var] += self.var_inc;
        if self.activity[var] > 1e100 {
            // Rescale all activities
            for a in self.activity.iter_mut() {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
    }

    /// Decay all variable activities.
    fn decay_var_activity(&mut self) {
        self.var_inc /= self.config.vsids_decay;
    }

    /// Solve the formula. Returns SAT with assignment or UNSAT with core.
    pub fn solve(&mut self) -> SatResult {
        let start = Instant::now();

        // Check for root-level conflicts detected during clause addition
        if let Some(conflict_ci) = self.root_conflict {
            self.stats.time_ms = start.elapsed().as_millis() as u64;
            return self.extract_unsat_core(conflict_ci);
        }

        let mut restart_count = 0u64;
        let mut conflicts_until_restart = match self.config.restart_strategy {
            crate::solver_config::RestartStrategy::None => u64::MAX,
            crate::solver_config::RestartStrategy::Fixed(n) => n,
            crate::solver_config::RestartStrategy::Luby(base) => {
                base * SolverConfig::luby_value(0)
            }
            crate::solver_config::RestartStrategy::Geometric { initial, .. } => initial,
        };
        let mut conflicts_since_restart = 0u64;

        // Initial propagation (for unit clauses added during setup)
        if let Some(conflict_ci) = self.propagate() {
            self.stats.time_ms = start.elapsed().as_millis() as u64;
            return self.extract_unsat_core(conflict_ci);
        }

        // After propagation, verify all unit clauses are satisfied
        for ci in 0..self.clauses.len() {
            let lits = &self.clauses[ci].lits;
            if lits.len() == 1 && self.lit_value(lits[0]) == LBool::False {
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return self.extract_unsat_core(ci);
            }
        }

        loop {
            // Check timeout
            if start.elapsed() > self.config.timeout {
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return SatResult::Unknown("timeout".to_string());
            }

            // Try to propagate
            if let Some(conflict_ci) = self.propagate() {
                self.stats.conflicts += 1;
                conflicts_since_restart += 1;

                if self.decision_level == 0 {
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    return self.extract_unsat_core(conflict_ci);
                }

                let (learned, bt_level) = self.analyze_conflict(conflict_ci);
                self.backtrack(bt_level);
                self.decay_var_activity();

                // After backtrack, enqueue the asserting literal
                if !learned.is_empty() {
                    let asserting_lit = learned[0];
                    let last_ci = self.clauses.len() - 1;
                    if self.lit_value(asserting_lit) == LBool::Undef {
                        self.enqueue(asserting_lit, last_ci);
                    }
                }

                // Restart check
                if conflicts_since_restart >= conflicts_until_restart {
                    restart_count += 1;
                    self.stats.restarts += 1;
                    self.backtrack(0);
                    conflicts_since_restart = 0;
                    conflicts_until_restart = match self.config.restart_strategy {
                        crate::solver_config::RestartStrategy::None => u64::MAX,
                        crate::solver_config::RestartStrategy::Fixed(n) => n,
                        crate::solver_config::RestartStrategy::Luby(base) => {
                            base * SolverConfig::luby_value(restart_count)
                        }
                        crate::solver_config::RestartStrategy::Geometric { initial, factor } => {
                            initial * (factor as u64).saturating_pow(restart_count as u32)
                        }
                    };
                }
            } else {
                // No conflict: make a decision
                if !self.decide() {
                    // All variables assigned => SAT
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    return SatResult::Sat(self.extract_assignment());
                }
            }
        }
    }

    /// Solve under a set of assumptions. Each assumption is a literal that must be true.
    pub fn solve_with_assumptions(&mut self, assumptions: &[Literal]) -> SatResult {
        let start = Instant::now();

        // Check root-level conflicts
        if let Some(conflict_ci) = self.root_conflict {
            self.stats.time_ms = start.elapsed().as_millis() as u64;
            return self.extract_unsat_core(conflict_ci);
        }

        // Initial propagation
        if let Some(conflict_ci) = self.propagate() {
            self.stats.time_ms = start.elapsed().as_millis() as u64;
            return self.extract_unsat_core(conflict_ci);
        }

        // Verify all unit clauses after propagation
        for ci in 0..self.clauses.len() {
            let lits = &self.clauses[ci].lits;
            if lits.len() == 1 && self.lit_value(lits[0]) == LBool::False {
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return self.extract_unsat_core(ci);
            }
        }

        // Assert assumptions at decision level 1..n
        for &assumption_lit in assumptions {
            self.decision_level += 1;
            self.trail_lim.push(self.trail.len());

            match self.lit_value(assumption_lit) {
                LBool::True => continue,
                LBool::False => {
                    // Assumption conflicts
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    self.backtrack(0);
                    return SatResult::Unsat(vec![vec![assumption_lit]]);
                }
                LBool::Undef => {
                    self.enqueue(assumption_lit, NO_REASON);
                    if let Some(conflict_ci) = self.propagate() {
                        let core = self.extract_unsat_core(conflict_ci);
                        self.backtrack(0);
                        self.stats.time_ms = start.elapsed().as_millis() as u64;
                        return core;
                    }
                }
            }
        }

        let assumption_level = self.decision_level;

        loop {
            if start.elapsed() > self.config.timeout {
                self.backtrack(0);
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return SatResult::Unknown("timeout".to_string());
            }

            if let Some(conflict_ci) = self.propagate() {
                self.stats.conflicts += 1;

                if self.decision_level <= assumption_level {
                    let core = self.extract_unsat_core(conflict_ci);
                    self.backtrack(0);
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    return core;
                }

                let (_learned, bt_level) = self.analyze_conflict(conflict_ci);
                let bt = bt_level.max(assumption_level);
                self.backtrack(bt);
                self.decay_var_activity();

                let last_ci = self.clauses.len() - 1;
                let asserting_lits = self.clauses[last_ci].lits.clone();
                if !asserting_lits.is_empty() {
                    let asserting_lit = asserting_lits[0];
                    if self.lit_value(asserting_lit) == LBool::Undef {
                        self.enqueue(asserting_lit, last_ci);
                    }
                }
            } else if !self.decide() {
                let result = SatResult::Sat(self.extract_assignment());
                self.backtrack(0);
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return result;
            }
        }
    }

    /// Extract the current assignment.
    fn extract_assignment(&self) -> Assignment {
        let mut assignment = Assignment::new();
        for v in 1..=self.num_vars {
            match self.assigns[v as usize] {
                LBool::True => assignment.set(v, true),
                LBool::False => assignment.set(v, false),
                LBool::Undef => assignment.set(v, false), // default
            }
        }
        assignment
    }

    /// Extract an UNSAT core from a conflict. Traces provenance to original clauses.
    fn extract_unsat_core(&self, conflict_ci: usize) -> SatResult {
        let mut core_original_indices: HashSet<usize> = HashSet::new();
        core_original_indices.extend(self.clauses[conflict_ci].provenance.iter());

        // Walk the trail to collect all reason clauses' provenances
        let mut seen = vec![false; self.num_vars as usize + 1];
        for &lit in &self.clauses[conflict_ci].lits {
            seen[lit_var(lit) as usize] = true;
        }

        for entry in self.trail.iter().rev() {
            let v = lit_var(entry.lit) as usize;
            if seen[v] && entry.reason != NO_REASON {
                core_original_indices.extend(self.clauses[entry.reason].provenance.iter());
                for &lit in &self.clauses[entry.reason].lits {
                    seen[lit_var(lit) as usize] = true;
                }
            }
        }

        let core_clauses: Vec<Clause> = core_original_indices
            .into_iter()
            .filter(|&idx| idx < self.clauses.len())
            .map(|idx| self.clauses[idx].lits.clone())
            .collect();

        SatResult::Unsat(core_clauses)
    }

    /// Get the number of variables.
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    /// Get the current decision level.
    pub fn current_level(&self) -> u32 {
        self.decision_level
    }

    /// Get clause count.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Get all clauses (for external inspection).
    pub fn get_clauses(&self) -> &[ClauseInfo] {
        &self.clauses
    }

    /// Reset the solver state (keep clauses, clear assignments).
    pub fn reset(&mut self) {
        let n = self.num_vars as usize + 1;
        self.assigns = vec![LBool::Undef; n];
        self.var_level = vec![0; n];
        self.var_reason = vec![NO_REASON; n];
        self.trail.clear();
        self.trail_lim.clear();
        self.decision_level = 0;
        self.prop_queue.clear();
        self.root_conflict = None;

        // Re-enqueue unit clauses from the clause database
        for ci in 0..self.clauses.len() {
            let lits = self.clauses[ci].lits.clone();
            if lits.len() == 1 {
                match self.lit_value(lits[0]) {
                    LBool::Undef => {
                        self.enqueue(lits[0], ci);
                    }
                    LBool::False => {
                        self.root_conflict = Some(ci);
                    }
                    LBool::True => {}
                }
            }
        }
    }
}

// ─── Convenience: solve a set of clauses directly ───────────────────────────

/// Solve a CNF formula given as a list of clauses.
pub fn solve_cnf(num_vars: u32, clauses: &[Clause]) -> SatResult {
    solve_cnf_with_config(num_vars, clauses, SolverConfig::default())
}

/// Solve a CNF formula with a custom configuration.
pub fn solve_cnf_with_config(num_vars: u32, clauses: &[Clause], config: SolverConfig) -> SatResult {
    let mut solver = DpllSolver::new(num_vars, config);
    for (i, clause) in clauses.iter().enumerate() {
        solver.add_original_clause(clause.clone(), i);
    }
    solver.solve()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolverConfig {
        SolverConfig::default()
    }

    #[test]
    fn test_empty_formula() {
        let result = solve_cnf(0, &[]);
        assert!(result.is_sat());
    }

    #[test]
    fn test_single_unit_clause() {
        // x1
        let result = solve_cnf(1, &[vec![1]]);
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        assert_eq!(a.get(1), Some(true));
    }

    #[test]
    fn test_contradictory_units() {
        // x1 AND NOT x1
        let result = solve_cnf(1, &[vec![1], vec![-1]]);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_simple_sat() {
        // (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2)
        let clauses = vec![vec![1, 2], vec![-1, 2], vec![1, -2]];
        let result = solve_cnf(2, &clauses);
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        // Must satisfy all clauses
        assert!(a.get(1) == Some(true) || a.get(2) == Some(true));
        assert!(a.get(1) == Some(false) || a.get(2) == Some(true));
        assert!(a.get(1) == Some(true) || a.get(2) == Some(false));
    }

    #[test]
    fn test_simple_unsat() {
        // (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2) AND (NOT x1 OR NOT x2)
        let clauses = vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]];
        let result = solve_cnf(2, &clauses);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_three_coloring_triangle() {
        // 3-coloring of a triangle (K3) is satisfiable with 3 colors
        // Variables: x_{v,c} for vertex v in {1,2,3}, color c in {1,2,3}
        // Variable numbering: (v-1)*3 + c, so v1c1=1, v1c2=2, v1c3=3, v2c1=4, ...
        let var = |v: i32, c: i32| -> i32 { (v - 1) * 3 + c };
        let mut clauses = Vec::new();

        // Each vertex has at least one color
        for v in 1..=3 {
            clauses.push(vec![var(v, 1), var(v, 2), var(v, 3)]);
        }
        // Adjacent vertices have different colors (for edges 1-2, 1-3, 2-3)
        for &(u, v) in &[(1, 2), (1, 3), (2, 3)] {
            for c in 1..=3 {
                clauses.push(vec![-var(u, c), -var(v, c)]);
            }
        }
        let result = solve_cnf(9, &clauses);
        assert!(result.is_sat());
    }

    #[test]
    fn test_pigeonhole_3_2() {
        // 3 pigeons, 2 holes: UNSAT
        // Variables: p_{i,j} = pigeon i in hole j
        let var = |i: i32, j: i32| -> i32 { (i - 1) * 2 + j };
        let mut clauses = Vec::new();

        // Each pigeon must be in some hole
        for i in 1..=3 {
            clauses.push(vec![var(i, 1), var(i, 2)]);
        }
        // No two pigeons in the same hole
        for j in 1..=2 {
            for i1 in 1..=3 {
                for i2 in (i1 + 1)..=3 {
                    clauses.push(vec![-var(i1, j), -var(i2, j)]);
                }
            }
        }
        let result = solve_cnf(6, &clauses);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_solve_with_assumptions_sat() {
        let mut solver = DpllSolver::new(3, default_config());
        // (x1 OR x2) AND (x2 OR x3)
        solver.add_original_clause(vec![1, 2], 0);
        solver.add_original_clause(vec![2, 3], 1);
        // Assume x1=false
        let result = solver.solve_with_assumptions(&[-1]);
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        assert_eq!(a.get(1), Some(false));
        // x2 must be true (forced by clause 0)
        assert_eq!(a.get(2), Some(true));
    }

    #[test]
    fn test_solve_with_assumptions_unsat() {
        let mut solver = DpllSolver::new(2, default_config());
        // (x1 OR x2) AND (NOT x2)
        solver.add_original_clause(vec![1, 2], 0);
        solver.add_original_clause(vec![-2], 1);
        // Assume x1=false => must have x2=true but -x2 is required => UNSAT
        let result = solver.solve_with_assumptions(&[-1]);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_learned_clause_speeds_up() {
        // Create a formula where learning helps: 4 variables
        // After learning, the solver should prune the search space
        let clauses = vec![
            vec![1, 2],
            vec![-1, 3],
            vec![-2, 3],
            vec![-3, 4],
            vec![-3, -4],
        ];
        let result = solve_cnf(4, &clauses);
        assert!(result.is_unsat());
    }

    #[test]
    fn test_statistics() {
        let clauses = vec![vec![1, 2], vec![-1, 2], vec![1, -2], vec![-1, -2]];
        let mut solver = DpllSolver::new(2, default_config());
        for (i, c) in clauses.iter().enumerate() {
            solver.add_original_clause(c.clone(), i);
        }
        solver.solve();
        assert!(solver.stats.conflicts > 0);
    }
}
