//! Fixpoint computation engines for abstract interpretation.
//!
//! Provides iterative solvers (worklist-based and chaotic iteration) that
//! compute least fixpoints of monotone equation systems over abstract domains.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Serialize, Deserialize};

use crate::lattice::Lattice;

// ---------------------------------------------------------------------------
// WorklistAlgorithm
// ---------------------------------------------------------------------------

/// Strategy used by the fixpoint engine to select the next node to process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorklistAlgorithm {
    /// Standard FIFO worklist.
    Fifo,
    /// LIFO (stack) worklist – often faster on reducible CFGs.
    Lifo,
    /// Priority-based: process nodes in reverse-postorder first.
    ReversePostOrder,
    /// Chaotic iteration without an explicit worklist.
    Chaotic,
}

impl Default for WorklistAlgorithm {
    fn default() -> Self {
        Self::ReversePostOrder
    }
}

// ---------------------------------------------------------------------------
// FixpointConfig
// ---------------------------------------------------------------------------

/// Tunables that govern fixpoint iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixpointConfig {
    /// Maximum number of iterations before the solver gives up.
    pub max_iterations: usize,
    /// Number of iterations before widening kicks in.
    pub widening_delay: usize,
    /// Number of narrowing iterations after the ascending chain stabilises.
    pub narrowing_iterations: usize,
    /// Worklist strategy.
    pub algorithm: WorklistAlgorithm,
    /// Whether to enable trace logging of each iteration step.
    pub trace_iterations: bool,
}

impl Default for FixpointConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            widening_delay: 3,
            narrowing_iterations: 2,
            algorithm: WorklistAlgorithm::default(),
            trace_iterations: false,
        }
    }
}

// ---------------------------------------------------------------------------
// FixpointResult
// ---------------------------------------------------------------------------

/// Outcome of a fixpoint computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixpointResult<V: Lattice + Serialize> {
    /// Per-block abstract values at the fixpoint.
    pub solution: BTreeMap<u64, V>,
    /// Total number of iterations performed.
    pub iterations: usize,
    /// Whether the computation converged within the iteration budget.
    pub converged: bool,
    /// Optional per-iteration diagnostics (populated when `trace_iterations` is set).
    pub iteration_trace: Vec<String>,
}

impl<V: Lattice + Serialize> FixpointResult<V> {
    /// Look up the fixpoint value for a given block id.
    pub fn get(&self, block: u64) -> Option<&V> {
        self.solution.get(&block)
    }
}

impl<V: Lattice + Serialize> Default for FixpointResult<V> {
    fn default() -> Self {
        Self {
            solution: BTreeMap::new(),
            iterations: 0,
            converged: false,
            iteration_trace: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// FixpointEngine
// ---------------------------------------------------------------------------

/// Iterative fixpoint solver parameterised over an abstract domain `V`.
#[derive(Debug, Clone)]
pub struct FixpointEngine<V: Lattice> {
    /// Solver configuration.
    pub config: FixpointConfig,
    /// Current per-node abstract state (keyed by block id).
    state: BTreeMap<u64, V>,
    /// Iteration counter.
    iteration: usize,
}

impl<V: Lattice + Serialize> FixpointEngine<V> {
    /// Create a new engine with the given configuration.
    pub fn new(config: FixpointConfig) -> Self {
        Self {
            config,
            state: BTreeMap::new(),
            iteration: usize::default(),
        }
    }

    /// Initialise a block with the given abstract value.
    pub fn set_initial(&mut self, block_id: u64, value: V) {
        self.state.insert(block_id, value);
    }

    /// Run the fixpoint iteration to convergence (or until the budget is
    /// exhausted) using the supplied transfer function.
    ///
    /// `transfer` receives a block id and its current abstract value, and must
    /// return the new abstract value.  `successors` maps a block id to its
    /// successor block ids.
    pub fn solve<F, S>(
        &mut self,
        mut transfer: F,
        _successors: S,
    ) -> FixpointResult<V>
    where
        F: FnMut(u64, &V) -> V,
        S: Fn(u64) -> Vec<u64>,
    {
        let mut converged = false;
        let mut trace = Vec::new();
        let block_ids: Vec<u64> = self.state.keys().copied().collect();

        for i in 0..self.config.max_iterations {
            self.iteration = i;
            let mut changed = false;

            for &bid in &block_ids {
                let current = match self.state.get(&bid) {
                    Some(v) => v.clone(),
                    None => continue,
                };
                let new_val = transfer(bid, &current);
                if !current.equivalent(&new_val) {
                    self.state.insert(bid, new_val);
                    changed = true;
                }
            }

            if self.config.trace_iterations {
                trace.push(format!("iteration {i}: changed={changed}"));
            }

            if !changed {
                converged = true;
                self.iteration = i + 1;
                break;
            }
        }

        FixpointResult {
            solution: self.state.clone(),
            iterations: self.iteration,
            converged,
            iteration_trace: trace,
        }
    }

    /// Return the current iteration count.
    pub fn iteration_count(&self) -> usize {
        self.iteration
    }
}

impl<V: Lattice + Serialize> Default for FixpointEngine<V> {
    fn default() -> Self {
        Self::new(FixpointConfig::default())
    }
}

impl<V: Lattice + Serialize> fmt::Display for FixpointResult<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FixpointResult(converged={}, iterations={}, nodes={})",
            self.converged,
            self.iterations,
            self.solution.len(),
        )
    }
}
