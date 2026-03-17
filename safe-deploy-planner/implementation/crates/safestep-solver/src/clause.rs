// Clause representation, clause database, watch lists, and clause management.

use crate::variable::{Assignment, DecisionLevel, Literal, LiteralVec, Variable};
use std::fmt;

// ── ClauseId ──────────────────────────────────────────────────────────────────

/// Index into the clause database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClauseId(pub u32);

impl ClauseId {
    pub fn new(index: u32) -> Self {
        ClauseId(index)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ClauseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.0)
    }
}

// ── ClauseStatus ──────────────────────────────────────────────────────────────

/// Evaluation status of a clause under a partial assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClauseStatus {
    /// At least one literal is true.
    Satisfied,
    /// Exactly one literal is unassigned; all others are false.
    Unit(Literal),
    /// All literals are false.
    Conflicting,
    /// More than one literal is unassigned.
    Unresolved,
}

// ── Clause ────────────────────────────────────────────────────────────────────

/// A disjunction of literals.
#[derive(Debug, Clone)]
pub struct Clause {
    /// The literals in this clause.
    pub literals: LiteralVec,
    /// Whether this is a learned clause (as opposed to an original clause).
    pub learned: bool,
    /// Activity score for clause deletion heuristic.
    pub activity: f64,
    /// Literal Block Distance (LBD / glue level).
    pub lbd: u32,
    /// Whether this clause has been marked for deletion.
    pub deleted: bool,
}

impl Clause {
    /// Create a new clause from literals.
    pub fn new(literals: LiteralVec) -> Self {
        Clause {
            literals,
            learned: false,
            activity: 0.0,
            lbd: 0,
            deleted: false,
        }
    }

    /// Create a new learned clause.
    pub fn new_learned(literals: LiteralVec, lbd: u32) -> Self {
        Clause {
            literals,
            learned: true,
            activity: 0.0,
            lbd,
            deleted: false,
        }
    }

    /// Number of literals.
    #[inline]
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Whether the clause is empty (a contradiction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Whether the clause is a unit clause (single literal).
    #[inline]
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Whether the clause is a binary clause.
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.literals.len() == 2
    }

    /// Get the first watched literal (index 0).
    #[inline]
    pub fn watch0(&self) -> Literal {
        self.literals[0]
    }

    /// Get the second watched literal (index 1).
    #[inline]
    pub fn watch1(&self) -> Literal {
        debug_assert!(self.literals.len() >= 2);
        self.literals[1]
    }

    /// Swap literals at positions i and j.
    pub fn swap_literals(&mut self, i: usize, j: usize) {
        self.literals.swap(i, j);
    }

    /// Evaluate the clause under the given assignment.
    pub fn status(&self, assignment: &Assignment) -> ClauseStatus {
        let mut unassigned_lit = None;
        let mut num_unassigned = 0;

        for &lit in &self.literals {
            match assignment.eval_literal(lit) {
                Some(true) => return ClauseStatus::Satisfied,
                Some(false) => {}
                None => {
                    num_unassigned += 1;
                    unassigned_lit = Some(lit);
                    if num_unassigned > 1 {
                        return ClauseStatus::Unresolved;
                    }
                }
            }
        }

        match num_unassigned {
            0 => ClauseStatus::Conflicting,
            1 => ClauseStatus::Unit(unassigned_lit.unwrap()),
            _ => ClauseStatus::Unresolved,
        }
    }

    /// Check if the clause is satisfied under the given assignment.
    pub fn is_satisfied(&self, assignment: &Assignment) -> bool {
        self.literals
            .iter()
            .any(|&lit| assignment.eval_literal(lit) == Some(true))
    }

    /// Compute the LBD (Literal Block Distance) of this clause under the given assignment.
    /// LBD = number of distinct decision levels among the literals.
    pub fn compute_lbd(&self, assignment: &Assignment) -> u32 {
        let mut levels = Vec::new();
        for &lit in &self.literals {
            let level = assignment.level(lit.var());
            if !levels.contains(&level) {
                levels.push(level);
            }
        }
        levels.len() as u32
    }

    /// Update the stored LBD value.
    pub fn update_lbd(&mut self, assignment: &Assignment) {
        self.lbd = self.compute_lbd(assignment);
    }

    /// Check if the clause contains a specific literal.
    pub fn contains(&self, lit: Literal) -> bool {
        self.literals.contains(&lit)
    }

    /// Check if the clause contains the variable (in either polarity).
    pub fn contains_var(&self, var: Variable) -> bool {
        self.literals.iter().any(|&l| l.var() == var)
    }

    /// Get the maximum decision level among literals in the clause.
    pub fn max_level(&self, assignment: &Assignment) -> DecisionLevel {
        self.literals
            .iter()
            .map(|&lit| assignment.level(lit.var()))
            .max()
            .unwrap_or(0)
    }

    /// Remove duplicate and tautological literals. Returns true if clause is tautological.
    pub fn simplify(&mut self) -> bool {
        self.literals.sort();
        self.literals.dedup();

        // Check for tautology: both x and ¬x present.
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i] == self.literals[j].negated() {
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, lit) in self.literals.iter().enumerate() {
            if i > 0 {
                write!(f, " ∨ ")?;
            }
            write!(f, "{}", lit)?;
        }
        write!(f, ")")
    }
}

// ── WatchList ─────────────────────────────────────────────────────────────────

/// Per-literal watch list entry: stores which clauses are watching this literal.
#[derive(Debug, Clone)]
pub struct WatchEntry {
    pub clause_id: ClauseId,
    /// The other watched literal in this clause (blocker for quick check).
    pub blocker: Literal,
}

/// Watch lists indexed by literal code.
#[derive(Debug, Clone)]
pub struct WatchList {
    watches: Vec<Vec<WatchEntry>>,
}

impl WatchList {
    /// Create a new watch list with capacity for the given number of literal codes.
    pub fn new(num_literal_codes: usize) -> Self {
        WatchList {
            watches: vec![Vec::new(); num_literal_codes],
        }
    }

    /// Add a watch entry for the given literal.
    pub fn add_watch(&mut self, lit: Literal, clause_id: ClauseId, blocker: Literal) {
        let code = lit.code();
        if code >= self.watches.len() {
            self.resize(code + 1);
        }
        self.watches[code].push(WatchEntry { clause_id, blocker });
    }

    /// Get the watch list for a literal.
    pub fn watches_for(&self, lit: Literal) -> &[WatchEntry] {
        let code = lit.code();
        if code < self.watches.len() {
            &self.watches[code]
        } else {
            &[]
        }
    }

    /// Get a mutable reference to the watch list for a literal.
    pub fn watches_for_mut(&mut self, lit: Literal) -> &mut Vec<WatchEntry> {
        let code = lit.code();
        if code >= self.watches.len() {
            self.resize(code + 1);
        }
        &mut self.watches[code]
    }

    /// Remove all watches for a clause from a literal's watch list.
    pub fn remove_clause_watches(&mut self, lit: Literal, clause_id: ClauseId) {
        let code = lit.code();
        if code < self.watches.len() {
            self.watches[code].retain(|w| w.clause_id != clause_id);
        }
    }

    /// Resize watch lists.
    pub fn resize(&mut self, new_size: usize) {
        if new_size > self.watches.len() {
            self.watches.resize(new_size, Vec::new());
        }
    }

    /// Clear all watch lists.
    pub fn clear(&mut self) {
        for wl in &mut self.watches {
            wl.clear();
        }
    }

    /// Total number of watch entries.
    pub fn total_watches(&self) -> usize {
        self.watches.iter().map(|w| w.len()).sum()
    }
}

// ── ClauseDatabase ────────────────────────────────────────────────────────────

/// Stores and manages all clauses.
#[derive(Debug, Clone)]
pub struct ClauseDatabase {
    /// All clauses (indexed by ClauseId).
    clauses: Vec<Clause>,
    /// Number of original (non-learned) clauses.
    num_original: usize,
    /// Number of learned clauses (not deleted).
    num_learned: usize,
    /// Free list of deleted clause indices for reuse.
    free_list: Vec<ClauseId>,
    /// Clause activity decay factor.
    clause_activity_increment: f64,
    /// Clause activity decay.
    clause_activity_decay: f64,
}

impl ClauseDatabase {
    /// Create a new empty clause database.
    pub fn new() -> Self {
        ClauseDatabase {
            clauses: Vec::new(),
            num_original: 0,
            num_learned: 0,
            free_list: Vec::new(),
            clause_activity_increment: 1.0,
            clause_activity_decay: 0.999,
        }
    }

    /// Add a clause to the database and return its id.
    pub fn add_clause(&mut self, clause: Clause) -> ClauseId {
        let learned = clause.learned;
        let id = if let Some(free_id) = self.free_list.pop() {
            self.clauses[free_id.index()] = clause;
            free_id
        } else {
            let id = ClauseId::new(self.clauses.len() as u32);
            self.clauses.push(clause);
            id
        };
        if learned {
            self.num_learned += 1;
        } else {
            self.num_original += 1;
        }
        id
    }

    /// Get a reference to a clause.
    #[inline]
    pub fn get(&self, id: ClauseId) -> &Clause {
        &self.clauses[id.index()]
    }

    /// Get a mutable reference to a clause.
    #[inline]
    pub fn get_mut(&mut self, id: ClauseId) -> &mut Clause {
        &mut self.clauses[id.index()]
    }

    /// Mark a clause as deleted and add it to the free list.
    pub fn remove_clause(&mut self, id: ClauseId) {
        let clause = &mut self.clauses[id.index()];
        if !clause.deleted {
            if clause.learned {
                self.num_learned -= 1;
            } else {
                self.num_original -= 1;
            }
            clause.deleted = true;
            self.free_list.push(id);
        }
    }

    /// Total number of active clauses.
    pub fn clause_count(&self) -> usize {
        self.num_original + self.num_learned
    }

    /// Number of original clauses.
    pub fn original_count(&self) -> usize {
        self.num_original
    }

    /// Number of learned clauses.
    pub fn learned_count(&self) -> usize {
        self.num_learned
    }

    /// Total capacity (including deleted slots).
    pub fn capacity(&self) -> usize {
        self.clauses.len()
    }

    /// Bump clause activity.
    pub fn bump_activity(&mut self, id: ClauseId) {
        self.clauses[id.index()].activity += self.clause_activity_increment;
        if self.clauses[id.index()].activity > 1e20 {
            for c in &mut self.clauses {
                c.activity *= 1e-20;
            }
            self.clause_activity_increment *= 1e-20;
        }
    }

    /// Apply clause activity decay.
    pub fn decay_activities(&mut self) {
        self.clause_activity_increment /= self.clause_activity_decay;
    }

    /// Garbage collect learned clauses based on activity and LBD thresholds.
    /// Returns the ids of deleted clauses.
    pub fn gc_learned_clauses(
        &mut self,
        max_learned: usize,
        activity_threshold: f64,
        max_lbd: u32,
    ) -> Vec<ClauseId> {
        if self.num_learned <= max_learned {
            return Vec::new();
        }

        let mut candidates: Vec<ClauseId> = (0..self.clauses.len())
            .filter_map(|i| {
                let c = &self.clauses[i];
                if c.learned && !c.deleted && c.lbd > 2 {
                    Some(ClauseId::new(i as u32))
                } else {
                    None
                }
            })
            .collect();

        // Sort by activity (ascending), so worst clauses are first.
        candidates.sort_by(|a, b| {
            let ca = &self.clauses[a.index()];
            let cb = &self.clauses[b.index()];
            ca.activity.partial_cmp(&cb.activity).unwrap()
        });

        let to_delete = self.num_learned.saturating_sub(max_learned / 2);
        let mut deleted = Vec::new();

        for &cid in candidates.iter().take(to_delete) {
            let c = &self.clauses[cid.index()];
            if c.activity < activity_threshold || c.lbd > max_lbd {
                deleted.push(cid);
            }
        }

        // If we didn't delete enough, delete more from the sorted list.
        if deleted.len() < to_delete {
            for &cid in candidates.iter().take(to_delete) {
                if !deleted.contains(&cid) {
                    deleted.push(cid);
                    if deleted.len() >= to_delete {
                        break;
                    }
                }
            }
        }

        for &cid in &deleted {
            self.remove_clause(cid);
        }

        deleted
    }

    /// Iterate over all active clause ids.
    pub fn active_clause_ids(&self) -> impl Iterator<Item = ClauseId> + '_ {
        (0..self.clauses.len()).filter_map(|i| {
            if !self.clauses[i].deleted {
                Some(ClauseId::new(i as u32))
            } else {
                None
            }
        })
    }

    /// Iterate over all active clauses with their ids.
    pub fn active_clauses(&self) -> impl Iterator<Item = (ClauseId, &Clause)> {
        self.clauses.iter().enumerate().filter_map(|(i, c)| {
            if !c.deleted {
                Some((ClauseId::new(i as u32), c))
            } else {
                None
            }
        })
    }

    /// Check if a clause id is valid and not deleted.
    pub fn is_active(&self, id: ClauseId) -> bool {
        let idx = id.index();
        idx < self.clauses.len() && !self.clauses[idx].deleted
    }
}

impl Default for ClauseDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::{Reason, Variable};
    use smallvec::smallvec;

    fn lit(v: i32) -> Literal {
        Literal::from_dimacs(v)
    }

    fn make_clause(dimacs: &[i32]) -> Clause {
        let lits: LiteralVec = dimacs.iter().map(|&d| Literal::from_dimacs(d)).collect();
        Clause::new(lits)
    }

    #[test]
    fn test_clause_basic() {
        let c = make_clause(&[1, -2, 3]);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert!(!c.is_unit());
        assert!(!c.is_binary());
    }

    #[test]
    fn test_clause_unit() {
        let c = make_clause(&[5]);
        assert!(c.is_unit());
        assert_eq!(c.literals[0], lit(5));
    }

    #[test]
    fn test_clause_empty() {
        let c = Clause::new(smallvec![]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_clause_binary() {
        let c = make_clause(&[1, -2]);
        assert!(c.is_binary());
    }

    #[test]
    fn test_clause_status_satisfied() {
        let c = make_clause(&[1, -2, 3]);
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision); // lit(1) = true
        assert_eq!(c.status(&asgn), ClauseStatus::Satisfied);
    }

    #[test]
    fn test_clause_status_unit() {
        let c = make_clause(&[1, -2, 3]);
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), false, 0, Reason::Decision); // lit(1) = false
        asgn.set(Variable::new(2), true, 0, Reason::Decision); // lit(-2) = false
        // lit(3) is unassigned -> unit
        assert_eq!(c.status(&asgn), ClauseStatus::Unit(lit(3)));
    }

    #[test]
    fn test_clause_status_conflicting() {
        let c = make_clause(&[1, -2]);
        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), false, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 0, Reason::Decision);
        assert_eq!(c.status(&asgn), ClauseStatus::Conflicting);
    }

    #[test]
    fn test_clause_status_unresolved() {
        let c = make_clause(&[1, 2, 3]);
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), false, 0, Reason::Decision);
        // 2, 3 unassigned
        assert_eq!(c.status(&asgn), ClauseStatus::Unresolved);
    }

    #[test]
    fn test_clause_is_satisfied() {
        let c = make_clause(&[1, -2]);
        let mut asgn = Assignment::new(2);
        assert!(!c.is_satisfied(&asgn));
        asgn.set(Variable::new(2), false, 0, Reason::Decision);
        assert!(c.is_satisfied(&asgn));
    }

    #[test]
    fn test_clause_contains() {
        let c = make_clause(&[1, -3, 5]);
        assert!(c.contains(lit(1)));
        assert!(c.contains(lit(-3)));
        assert!(!c.contains(lit(2)));
        assert!(c.contains_var(Variable::new(3)));
        assert!(!c.contains_var(Variable::new(4)));
    }

    #[test]
    fn test_clause_simplify_dedup() {
        let mut c = Clause::new(smallvec![lit(1), lit(2), lit(1)]);
        let taut = c.simplify();
        assert!(!taut);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_clause_simplify_tautology() {
        let mut c = Clause::new(smallvec![lit(1), lit(-1), lit(2)]);
        let taut = c.simplify();
        assert!(taut);
    }

    #[test]
    fn test_clause_lbd() {
        let c = make_clause(&[1, -2, 3]);
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), true, 1, Reason::Decision);
        asgn.set(Variable::new(3), true, 2, Reason::Decision);
        assert_eq!(c.compute_lbd(&asgn), 3);
    }

    #[test]
    fn test_clause_max_level() {
        let c = make_clause(&[1, -2, 3]);
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 2, Reason::Decision);
        asgn.set(Variable::new(2), true, 5, Reason::Decision);
        asgn.set(Variable::new(3), true, 3, Reason::Decision);
        assert_eq!(c.max_level(&asgn), 5);
    }

    #[test]
    fn test_clause_display() {
        let c = make_clause(&[1, -2]);
        let s = format!("{}", c);
        assert!(s.contains("x1"));
        assert!(s.contains("x2"));
    }

    #[test]
    fn test_clause_watched() {
        let c = make_clause(&[1, -2, 3]);
        assert_eq!(c.watch0(), lit(1));
        assert_eq!(c.watch1(), lit(-2));
    }

    // ── ClauseDatabase Tests ──────────────────────────────────────────────────

    #[test]
    fn test_clause_db_add_and_get() {
        let mut db = ClauseDatabase::new();
        let c = make_clause(&[1, -2, 3]);
        let id = db.add_clause(c);
        assert_eq!(db.clause_count(), 1);
        assert_eq!(db.get(id).len(), 3);
    }

    #[test]
    fn test_clause_db_remove() {
        let mut db = ClauseDatabase::new();
        let c = make_clause(&[1, -2]);
        let id = db.add_clause(c);
        assert_eq!(db.clause_count(), 1);
        db.remove_clause(id);
        assert_eq!(db.clause_count(), 0);
        assert!(!db.is_active(id));
    }

    #[test]
    fn test_clause_db_reuse_deleted() {
        let mut db = ClauseDatabase::new();
        let c1 = make_clause(&[1, -2]);
        let id1 = db.add_clause(c1);
        db.remove_clause(id1);

        let c2 = make_clause(&[3, -4]);
        let id2 = db.add_clause(c2);
        // Should reuse the freed slot.
        assert_eq!(id2, id1);
        assert_eq!(db.clause_count(), 1);
    }

    #[test]
    fn test_clause_db_learned_count() {
        let mut db = ClauseDatabase::new();
        let c1 = make_clause(&[1, -2]);
        db.add_clause(c1);
        let c2 = Clause::new_learned(smallvec![lit(3), lit(-4)], 2);
        db.add_clause(c2);
        assert_eq!(db.original_count(), 1);
        assert_eq!(db.learned_count(), 1);
    }

    #[test]
    fn test_clause_db_bump_activity() {
        let mut db = ClauseDatabase::new();
        let c = make_clause(&[1, 2]);
        let id = db.add_clause(c);
        db.bump_activity(id);
        assert!(db.get(id).activity > 0.0);
    }

    #[test]
    fn test_clause_db_gc() {
        let mut db = ClauseDatabase::new();
        // Add many learned clauses.
        for i in 1..=20 {
            let mut c = Clause::new_learned(smallvec![lit(i), lit(-(i + 1))], 10);
            c.activity = 0.001;
            db.add_clause(c);
        }
        assert_eq!(db.learned_count(), 20);
        let deleted = db.gc_learned_clauses(10, 1e-8, 30);
        assert!(!deleted.is_empty());
        assert!(db.learned_count() <= 20);
    }

    #[test]
    fn test_clause_db_active_ids() {
        let mut db = ClauseDatabase::new();
        let id1 = db.add_clause(make_clause(&[1, 2]));
        let id2 = db.add_clause(make_clause(&[3, 4]));
        let _id3 = db.add_clause(make_clause(&[5, 6]));
        db.remove_clause(id2);
        let active: Vec<_> = db.active_clause_ids().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&id1));
    }

    // ── WatchList Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_watchlist_basic() {
        let mut wl = WatchList::new(20);
        let l = lit(1);
        wl.add_watch(l, ClauseId::new(0), lit(2));
        assert_eq!(wl.watches_for(l).len(), 1);
        assert_eq!(wl.watches_for(l)[0].clause_id, ClauseId::new(0));
    }

    #[test]
    fn test_watchlist_remove() {
        let mut wl = WatchList::new(20);
        let l = lit(1);
        wl.add_watch(l, ClauseId::new(0), lit(2));
        wl.add_watch(l, ClauseId::new(1), lit(3));
        wl.remove_clause_watches(l, ClauseId::new(0));
        assert_eq!(wl.watches_for(l).len(), 1);
        assert_eq!(wl.watches_for(l)[0].clause_id, ClauseId::new(1));
    }

    #[test]
    fn test_watchlist_clear() {
        let mut wl = WatchList::new(20);
        wl.add_watch(lit(1), ClauseId::new(0), lit(2));
        wl.add_watch(lit(2), ClauseId::new(1), lit(1));
        wl.clear();
        assert_eq!(wl.total_watches(), 0);
    }

    #[test]
    fn test_watchlist_resize() {
        let mut wl = WatchList::new(4);
        // Adding a watch for a literal beyond current size should auto-resize.
        wl.add_watch(Literal::from_code(100), ClauseId::new(0), lit(1));
        assert_eq!(wl.watches_for(Literal::from_code(100)).len(), 1);
    }

    #[test]
    fn test_clause_swap_literals() {
        let mut c = make_clause(&[1, -2, 3]);
        c.swap_literals(0, 2);
        assert_eq!(c.literals[0], lit(3));
        assert_eq!(c.literals[2], lit(1));
    }
}
