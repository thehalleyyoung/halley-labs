// regsynth-solver: solver configuration
// Timeout settings, heuristic parameters, solver selection, and tuning knobs.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Which solver backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverBackend {
    Sat,
    Smt,
    MaxSmt,
    Ilp,
}

impl Default for SolverBackend {
    fn default() -> Self {
        Self::Smt
    }
}

/// Restart strategy for the SAT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RestartStrategy {
    /// No restarts.
    None,
    /// Fixed interval restarts.
    Fixed(u64),
    /// Luby sequence restarts with a base unit.
    Luby(u64),
    /// Geometric growth: multiply interval by factor after each restart.
    Geometric { initial: u64, factor: u32 },
}

impl Default for RestartStrategy {
    fn default() -> Self {
        Self::Luby(100)
    }
}

/// Variable decision heuristic for the SAT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionHeuristic {
    /// VSIDS (Variable State Independent Decaying Sum).
    Vsids,
    /// Random variable selection.
    Random,
    /// Choose the variable that appears most frequently in clauses.
    MostFrequent,
}

impl Default for DecisionHeuristic {
    fn default() -> Self {
        Self::Vsids
    }
}

/// Clause deletion policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClauseDeletionPolicy {
    /// Keep all learned clauses.
    KeepAll,
    /// Delete clauses exceeding activity threshold.
    ActivityBased {
        max_learned: usize,
        decay_factor: f64,
    },
    /// Delete clauses exceeding a length threshold.
    LengthBased { max_length: usize },
}

impl Default for ClauseDeletionPolicy {
    fn default() -> Self {
        Self::ActivityBased {
            max_learned: 10_000,
            decay_factor: 0.95,
        }
    }
}

/// Branch-and-bound node selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeSelection {
    BestFirst,
    DepthFirst,
    BreadthFirst,
}

impl Default for NodeSelection {
    fn default() -> Self {
        Self::BestFirst
    }
}

/// Variable branching strategy for ILP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchingStrategy {
    MostFractional,
    FirstFractional,
    StrongBranching,
}

impl Default for BranchingStrategy {
    fn default() -> Self {
        Self::MostFractional
    }
}

/// Verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Verbosity {
    Silent,
    Errors,
    Warnings,
    Info,
    Debug,
    Trace,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::Warnings
    }
}

/// Complete solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Solver backend to use.
    pub backend: SolverBackend,

    /// Maximum wall-clock time.
    pub timeout: Duration,

    /// Maximum memory (bytes). 0 = unlimited.
    pub memory_limit: usize,

    /// Random seed for reproducibility.
    pub random_seed: u64,

    /// Verbosity level.
    pub verbosity: Verbosity,

    /// SAT solver: restart strategy.
    pub restart_strategy: RestartStrategy,

    /// SAT solver: decision heuristic.
    pub decision_heuristic: DecisionHeuristic,

    /// SAT solver: VSIDS decay factor (0.0–1.0, typically ~0.95).
    pub vsids_decay: f64,

    /// SAT solver: phase saving (reuse polarity from last assignment).
    pub phase_saving: bool,

    /// SAT solver: clause deletion policy.
    pub clause_deletion: ClauseDeletionPolicy,

    /// ILP: node selection strategy for branch-and-bound.
    pub node_selection: NodeSelection,

    /// ILP: variable branching strategy.
    pub branching_strategy: BranchingStrategy,

    /// ILP: absolute MIP gap tolerance.
    pub mip_gap_absolute: f64,

    /// ILP: relative MIP gap tolerance.
    pub mip_gap_relative: f64,

    /// Pareto: epsilon for approximate coverage.
    pub pareto_epsilon: f64,

    /// Pareto: maximum number of points to enumerate.
    pub pareto_max_points: usize,

    /// Pareto: number of scalarization directions to sample.
    pub pareto_num_directions: usize,

    /// MaxSMT: maximum number of Fu-Malik iterations.
    pub maxsmt_max_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            backend: SolverBackend::default(),
            timeout: Duration::from_secs(300),
            memory_limit: 0,
            random_seed: 42,
            verbosity: Verbosity::default(),
            restart_strategy: RestartStrategy::default(),
            decision_heuristic: DecisionHeuristic::default(),
            vsids_decay: 0.95,
            phase_saving: true,
            clause_deletion: ClauseDeletionPolicy::default(),
            node_selection: NodeSelection::default(),
            branching_strategy: BranchingStrategy::default(),
            mip_gap_absolute: 1e-6,
            mip_gap_relative: 1e-4,
            pareto_epsilon: 0.01,
            pareto_max_points: 100,
            pareto_num_directions: 20,
            maxsmt_max_iterations: 1000,
        }
    }
}

impl SolverConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_backend(mut self, backend: SolverBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    pub fn with_verbosity(mut self, v: Verbosity) -> Self {
        self.verbosity = v;
        self
    }

    pub fn with_pareto_epsilon(mut self, eps: f64) -> Self {
        self.pareto_epsilon = eps;
        self
    }

    pub fn with_pareto_max_points(mut self, n: usize) -> Self {
        self.pareto_max_points = n;
        self
    }

    /// Compute the Luby sequence value at index i (0-indexed).
    /// Sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    pub fn luby_value(i: u64) -> u64 {
        // MiniSat-style Luby computation
        let mut size = 1u64;
        let mut seq = 0u32;
        while size < i + 1 {
            seq += 1;
            size = 2 * size + 1;
        }
        let mut x = i;
        while size - 1 != x {
            size = (size - 1) / 2;
            seq -= 1;
            x %= size;
        }
        1u64 << seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.backend, SolverBackend::Smt);
        assert_eq!(cfg.timeout, Duration::from_secs(300));
        assert_eq!(cfg.random_seed, 42);
        assert!((cfg.vsids_decay - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_builder_pattern() {
        let cfg = SolverConfig::new()
            .with_timeout(Duration::from_secs(60))
            .with_backend(SolverBackend::MaxSmt)
            .with_seed(123);
        assert_eq!(cfg.backend, SolverBackend::MaxSmt);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
        assert_eq!(cfg.random_seed, 123);
    }

    #[test]
    fn test_luby_sequence() {
        // Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        assert_eq!(SolverConfig::luby_value(0), 1);
        assert_eq!(SolverConfig::luby_value(1), 1);
        assert_eq!(SolverConfig::luby_value(2), 2);
        assert_eq!(SolverConfig::luby_value(3), 1);
        assert_eq!(SolverConfig::luby_value(4), 1);
        assert_eq!(SolverConfig::luby_value(5), 2);
        assert_eq!(SolverConfig::luby_value(6), 4);
    }

    #[test]
    fn test_verbosity_ordering() {
        assert!(Verbosity::Silent < Verbosity::Errors);
        assert!(Verbosity::Errors < Verbosity::Debug);
        assert!(Verbosity::Debug < Verbosity::Trace);
    }
}
