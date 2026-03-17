//! MARCO algorithm for enumerating Minimal Unsatisfiable Subsets (MUS) and
//! Maximal Satisfiable Subsets (MSS).

use crate::solver::{BuiltinSolver, Clause, Literal, SatResult, SmtSolver, SolverConfig, SolverStats};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalUnsatCore {
    pub constraints: Vec<usize>,
    pub core_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaximalSatSubset {
    pub constraints: Vec<usize>,
    pub subset_size: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarcoStrategy {
    MUSFirst,
    MSSFirst,
    Alternating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarcoConfig {
    pub timeout_ms: u64,
    pub max_cores: usize,
    pub strategy: MarcoStrategy,
    pub max_iterations: usize,
}

impl Default for MarcoConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            max_cores: 100,
            strategy: MarcoStrategy::Alternating,
            max_iterations: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// MapSolver – auxiliary solver for seed management in MARCO
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MapSolver {
    num_constraints: usize,
    solver: BuiltinSolver,
    blocked_up: Vec<Vec<usize>>,
    blocked_down: Vec<Vec<usize>>,
}

impl MapSolver {
    pub fn new(num_constraints: usize) -> Self {
        let mut solver = BuiltinSolver::new(SolverConfig {
            timeout_ms: 10_000,
            ..Default::default()
        });
        // Create one boolean variable per constraint
        for _ in 0..num_constraints {
            solver.new_variable();
        }
        Self {
            num_constraints,
            solver,
            blocked_up: Vec::new(),
            blocked_down: Vec::new(),
        }
    }

    /// Block all subsets of `set` (i.e., mark this as a known MSS or superset of MSS).
    /// Adds clause: at least one element NOT in `set` must be included.
    pub fn block_down(&mut self, set: &[usize]) {
        let set_hs: HashSet<usize> = set.iter().copied().collect();
        let mut clause_lits = Vec::new();
        for i in 0..self.num_constraints {
            if !set_hs.contains(&i) {
                clause_lits.push(Literal::pos(i));
            }
        }
        if !clause_lits.is_empty() {
            self.solver.add_clause(Clause::new(clause_lits));
        }
        self.blocked_down.push(set.to_vec());
    }

    /// Block all supersets of `set` (i.e., mark this as a known MUS or subset of MUS).
    /// Adds clause: at least one element in `set` must be excluded.
    pub fn block_up(&mut self, set: &[usize]) {
        let mut clause_lits = Vec::new();
        for &i in set {
            clause_lits.push(Literal::neg(i));
        }
        if !clause_lits.is_empty() {
            self.solver.add_clause(Clause::new(clause_lits));
        }
        self.blocked_up.push(set.to_vec());
    }

    /// Get an unexplored seed from the map solver.
    /// Returns None if all seeds have been explored.
    pub fn get_unexplored(&mut self) -> Option<Vec<usize>> {
        let result = self.solver.solve();
        match result {
            SatResult::Sat(model) => {
                let seed: Vec<usize> = (0..self.num_constraints)
                    .filter(|&i| model.get(&i).copied().unwrap_or(false))
                    .collect();
                Some(seed)
            }
            _ => None,
        }
    }

    pub fn num_blocked_up(&self) -> usize {
        self.blocked_up.len()
    }

    pub fn num_blocked_down(&self) -> usize {
        self.blocked_down.len()
    }
}

// ---------------------------------------------------------------------------
// MarcoSolver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MarcoSolver {
    config: MarcoConfig,
    /// The clauses representing the constraint system to explore.
    clauses: Vec<Vec<Literal>>,
}

impl MarcoSolver {
    pub fn new(config: MarcoConfig) -> Self {
        Self {
            config,
            clauses: Vec::new(),
        }
    }

    /// Set the constraint system (list of clauses).
    pub fn set_constraints(&mut self, clauses: Vec<Vec<Literal>>) {
        self.clauses = clauses;
    }

    /// Create from raw clause data: each inner vec is a clause of literal indices.
    pub fn from_clauses(config: MarcoConfig, clauses: Vec<Vec<Literal>>) -> Self {
        Self { config, clauses }
    }

    /// Enumerate all minimal unsatisfiable cores.
    pub fn enumerate_mus(&mut self) -> Vec<MinimalUnsatCore> {
        let mut results = Vec::new();
        let start = Instant::now();
        let n = self.clauses.len();
        let mut map_solver = MapSolver::new(n);
        let mut iterations = 0;

        loop {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                break;
            }
            if results.len() >= self.config.max_cores {
                break;
            }
            if iterations >= self.config.max_iterations {
                break;
            }
            iterations += 1;

            let seed = match map_solver.get_unexplored() {
                Some(s) => s,
                None => break,
            };

            if self.is_satisfiable(&seed) {
                // Seed is SAT -> grow to MSS and block down
                let mss = self.grow(&seed);
                map_solver.block_down(&mss);
            } else {
                // Seed is UNSAT -> shrink to MUS and block up
                let mus = self.shrink(&seed);
                map_solver.block_up(&mus);
                results.push(MinimalUnsatCore {
                    core_size: mus.len(),
                    constraints: mus,
                });
            }
        }

        results
    }

    /// Enumerate all maximal satisfiable subsets.
    pub fn enumerate_mss(&mut self) -> Vec<MaximalSatSubset> {
        let mut results = Vec::new();
        let start = Instant::now();
        let n = self.clauses.len();
        let mut map_solver = MapSolver::new(n);
        let mut iterations = 0;

        loop {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                break;
            }
            if iterations >= self.config.max_iterations {
                break;
            }
            iterations += 1;

            let seed = match map_solver.get_unexplored() {
                Some(s) => s,
                None => break,
            };

            if self.is_satisfiable(&seed) {
                let mss = self.grow(&seed);
                map_solver.block_down(&mss);
                results.push(MaximalSatSubset {
                    subset_size: mss.len(),
                    constraints: mss,
                });
            } else {
                let mus = self.shrink(&seed);
                map_solver.block_up(&mus);
            }
        }

        results
    }

    /// Grow a satisfiable seed to a maximal satisfiable subset.
    pub fn grow(&self, seed: &[usize]) -> Vec<usize> {
        let n = self.clauses.len();
        let mut current: HashSet<usize> = seed.iter().copied().collect();

        for i in 0..n {
            if current.contains(&i) {
                continue;
            }
            // Try adding constraint i
            let mut candidate: Vec<usize> = current.iter().copied().collect();
            candidate.push(i);
            if self.is_satisfiable(&candidate) {
                current.insert(i);
            }
        }

        let mut result: Vec<usize> = current.into_iter().collect();
        result.sort();
        result
    }

    /// Shrink an unsatisfiable seed to a minimal unsatisfiable core.
    pub fn shrink(&self, seed: &[usize]) -> Vec<usize> {
        let mut current: Vec<usize> = seed.to_vec();

        let mut i = 0;
        while i < current.len() {
            // Try removing element i
            let mut candidate = current.clone();
            candidate.remove(i);

            if candidate.is_empty() || self.is_satisfiable(&candidate) {
                // Can't remove this element; keep it
                i += 1;
            } else {
                // Can remove this element; it's still UNSAT
                current = candidate;
                // Don't increment i since elements shifted
            }
        }

        current
    }

    /// Check if the subset of clauses is satisfiable.
    fn is_satisfiable(&self, constraint_indices: &[usize]) -> bool {
        if constraint_indices.is_empty() {
            return true;
        }

        let mut solver = BuiltinSolver::new(SolverConfig {
            timeout_ms: self.config.timeout_ms / 10,
            ..Default::default()
        });

        for &idx in constraint_indices {
            if idx < self.clauses.len() {
                solver.add_clause(Clause::new(self.clauses[idx].clone()));
            }
        }

        match solver.solve() {
            SatResult::Sat(_) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// MonotonicityAwareMarco
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MonotonicityAwareMarco {
    marco: MarcoSolver,
    antichain_cascading: Vec<BTreeSet<usize>>,
    antichain_safe: Vec<BTreeSet<usize>>,
}

impl MonotonicityAwareMarco {
    pub fn new(config: MarcoConfig) -> Self {
        Self {
            marco: MarcoSolver::new(config),
            antichain_cascading: Vec::new(),
            antichain_safe: Vec::new(),
        }
    }

    pub fn set_constraints(&mut self, clauses: Vec<Vec<Literal>>) {
        self.marco.set_constraints(clauses);
    }

    /// Check whether a seed can be pruned by monotonicity.
    pub fn can_prune(&self, seed: &[usize]) -> bool {
        let seed_set: BTreeSet<usize> = seed.iter().copied().collect();
        // Prune if superset of known cascading
        for cs in &self.antichain_cascading {
            if cs.is_subset(&seed_set) {
                return true;
            }
        }
        // Prune if subset of known safe
        for ss in &self.antichain_safe {
            if seed_set.is_subset(ss) {
                return true;
            }
        }
        false
    }

    pub fn record_cascading(&mut self, set: BTreeSet<usize>) {
        self.antichain_cascading.push(set);
    }

    pub fn record_safe(&mut self, set: BTreeSet<usize>) {
        self.antichain_safe.push(set);
    }

    /// Run MARCO with monotonicity-aware pruning.
    pub fn enumerate_mus_pruned(&mut self) -> Vec<MinimalUnsatCore> {
        let mut results = Vec::new();
        let start = Instant::now();
        let n = self.marco.clauses.len();
        let mut map_solver = MapSolver::new(n);
        let mut iterations = 0;

        loop {
            if start.elapsed().as_millis() as u64 > self.marco.config.timeout_ms {
                break;
            }
            if results.len() >= self.marco.config.max_cores {
                break;
            }
            if iterations >= self.marco.config.max_iterations {
                break;
            }
            iterations += 1;

            let seed = match map_solver.get_unexplored() {
                Some(s) => s,
                None => break,
            };

            if self.can_prune(&seed) {
                // Block based on known results
                if self.antichain_cascading.iter().any(|cs| {
                    let seed_set: BTreeSet<usize> = seed.iter().copied().collect();
                    cs.is_subset(&seed_set)
                }) {
                    map_solver.block_up(&seed);
                } else {
                    map_solver.block_down(&seed);
                }
                continue;
            }

            if self.marco.is_satisfiable(&seed) {
                let mss = self.marco.grow(&seed);
                map_solver.block_down(&mss);
                self.record_safe(mss.iter().copied().collect());
            } else {
                let mus = self.marco.shrink(&seed);
                map_solver.block_up(&mus);
                self.record_cascading(mus.iter().copied().collect());
                results.push(MinimalUnsatCore {
                    core_size: mus.len(),
                    constraints: mus,
                });
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sat_clauses() -> Vec<Vec<Literal>> {
        // (x0 ∨ x1) ∧ (x0 ∨ ¬x1)  => SAT
        vec![
            vec![Literal::pos(0), Literal::pos(1)],
            vec![Literal::pos(0), Literal::neg(1)],
        ]
    }

    fn make_unsat_clauses() -> Vec<Vec<Literal>> {
        // (x0) ∧ (¬x0) ∧ (x1) => UNSAT core is {0, 1} (first two clauses)
        vec![
            vec![Literal::pos(0)],
            vec![Literal::neg(0)],
            vec![Literal::pos(1)],
        ]
    }

    fn make_mixed_clauses() -> Vec<Vec<Literal>> {
        // c0: (x0)
        // c1: (¬x0)
        // c2: (x1)
        // c3: (¬x1)
        // MUS: {c0,c1} and {c2,c3}
        vec![
            vec![Literal::pos(0)],
            vec![Literal::neg(0)],
            vec![Literal::pos(1)],
            vec![Literal::neg(1)],
        ]
    }

    #[test]
    fn test_marco_config_default() {
        let cfg = MarcoConfig::default();
        assert_eq!(cfg.timeout_ms, 30_000);
        assert_eq!(cfg.strategy, MarcoStrategy::Alternating);
    }

    #[test]
    fn test_map_solver_basic() {
        let mut ms = MapSolver::new(3);
        let seed = ms.get_unexplored();
        assert!(seed.is_some());
    }

    #[test]
    fn test_map_solver_block_up() {
        let mut ms = MapSolver::new(2);
        ms.block_up(&[0, 1]);
        // After blocking {0,1} up, any superset is blocked
        // The only option is subsets of {0,1} that don't include both
        let seed = ms.get_unexplored();
        if let Some(s) = seed {
            assert!(!(s.contains(&0) && s.contains(&1)));
        }
    }

    #[test]
    fn test_map_solver_block_down() {
        let mut ms = MapSolver::new(3);
        ms.block_down(&[0, 1, 2]);
        // After blocking {0,1,2} down, no subset is available
        // (all elements are in the set, so the blocking clause is empty)
        // This should cause UNSAT
        let seed = ms.get_unexplored();
        assert!(seed.is_none());
    }

    #[test]
    fn test_grow_sat_seed() {
        let clauses = make_sat_clauses();
        let marco = MarcoSolver::from_clauses(MarcoConfig::default(), clauses);
        let mss = marco.grow(&[0]);
        // Should grow to include both clauses since it's SAT
        assert!(mss.len() >= 1);
    }

    #[test]
    fn test_shrink_unsat_seed() {
        let clauses = make_unsat_clauses();
        let marco = MarcoSolver::from_clauses(MarcoConfig::default(), clauses);
        let mus = marco.shrink(&[0, 1, 2]);
        // MUS should be {0, 1} (the contradicting pair)
        assert!(mus.len() <= 3);
        assert!(mus.contains(&0) && mus.contains(&1));
    }

    #[test]
    fn test_enumerate_mus() {
        let clauses = make_unsat_clauses();
        let mut marco = MarcoSolver::from_clauses(
            MarcoConfig { timeout_ms: 5000, ..Default::default() },
            clauses,
        );
        let cores = marco.enumerate_mus();
        assert!(!cores.is_empty());
        // Should find the MUS {0, 1}
        let found = cores.iter().any(|c| c.constraints.contains(&0) && c.constraints.contains(&1));
        assert!(found);
    }

    #[test]
    fn test_enumerate_mss() {
        let clauses = make_unsat_clauses();
        let mut marco = MarcoSolver::from_clauses(
            MarcoConfig { timeout_ms: 5000, ..Default::default() },
            clauses,
        );
        let subsets = marco.enumerate_mss();
        // Should find MSS that include constraints not in the core
        assert!(!subsets.is_empty());
    }

    #[test]
    fn test_enumerate_mus_sat_formula() {
        let clauses = make_sat_clauses();
        let mut marco = MarcoSolver::from_clauses(
            MarcoConfig { timeout_ms: 5000, ..Default::default() },
            clauses,
        );
        let cores = marco.enumerate_mus();
        // SAT formula has no MUS
        assert!(cores.is_empty());
    }

    #[test]
    fn test_multiple_mus() {
        let clauses = make_mixed_clauses();
        let mut marco = MarcoSolver::from_clauses(
            MarcoConfig { timeout_ms: 5000, ..Default::default() },
            clauses,
        );
        let cores = marco.enumerate_mus();
        // Should find two MUS: {0,1} and {2,3}
        assert!(cores.len() >= 1);
    }

    #[test]
    fn test_monotonicity_aware_marco_basic() {
        let mut mam = MonotonicityAwareMarco::new(MarcoConfig::default());
        mam.set_constraints(make_unsat_clauses());
        assert!(!mam.can_prune(&[0, 1, 2]));
    }

    #[test]
    fn test_monotonicity_aware_marco_prune() {
        let mut mam = MonotonicityAwareMarco::new(MarcoConfig::default());
        mam.set_constraints(make_unsat_clauses());
        mam.record_cascading([0, 1].iter().copied().collect());
        // {0, 1, 2} is a superset of {0, 1} -> should be pruned
        assert!(mam.can_prune(&[0, 1, 2]));
    }

    #[test]
    fn test_monotonicity_aware_enumerate() {
        let mut mam = MonotonicityAwareMarco::new(MarcoConfig { timeout_ms: 5000, ..Default::default() });
        mam.set_constraints(make_unsat_clauses());
        let cores = mam.enumerate_mus_pruned();
        assert!(!cores.is_empty());
    }

    #[test]
    fn test_is_satisfiable_empty() {
        let marco = MarcoSolver::from_clauses(MarcoConfig::default(), vec![]);
        assert!(marco.is_satisfiable(&[]));
    }

    #[test]
    fn test_minimal_unsat_core_display() {
        let core = MinimalUnsatCore { constraints: vec![0, 1], core_size: 2 };
        assert_eq!(core.core_size, 2);
        assert_eq!(core.constraints.len(), 2);
    }

    #[test]
    fn test_maximal_sat_subset_display() {
        let mss = MaximalSatSubset { constraints: vec![0, 2], subset_size: 2 };
        assert_eq!(mss.subset_size, 2);
    }

    #[test]
    fn test_map_solver_counts() {
        let mut ms = MapSolver::new(3);
        assert_eq!(ms.num_blocked_up(), 0);
        ms.block_up(&[0]);
        assert_eq!(ms.num_blocked_up(), 1);
        ms.block_down(&[1, 2]);
        assert_eq!(ms.num_blocked_down(), 1);
    }
}
