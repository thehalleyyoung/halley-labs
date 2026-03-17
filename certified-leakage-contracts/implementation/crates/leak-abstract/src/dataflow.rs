//! Classical monotone-framework dataflow analysis.
//!
//! Wires together a control-flow graph, transfer functions, and the fixpoint
//! solver to compute per-program-point abstract values.

use std::collections::BTreeMap;

use serde::{Serialize, Deserialize};

use crate::fixpoint::{FixpointConfig, FixpointEngine, FixpointResult};
use crate::lattice::Lattice;

// ---------------------------------------------------------------------------
// DataflowResult
// ---------------------------------------------------------------------------

/// The result of running a dataflow analysis over a CFG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowResult<V: Lattice + Serialize> {
    /// Abstract value at the *entry* of each basic block.
    pub entry_states: BTreeMap<u64, V>,
    /// Abstract value at the *exit* of each basic block.
    pub exit_states: BTreeMap<u64, V>,
    /// Number of fixpoint iterations performed.
    pub iterations: usize,
    /// Whether the analysis converged.
    pub converged: bool,
}

impl<V: Lattice + Serialize> DataflowResult<V> {
    /// Get the abstract value at the entry of a block.
    pub fn entry_state(&self, block: u64) -> Option<&V> {
        self.entry_states.get(&block)
    }

    /// Get the abstract value at the exit of a block.
    pub fn exit_state(&self, block: u64) -> Option<&V> {
        self.exit_states.get(&block)
    }
}

impl<V: Lattice + Serialize> Default for DataflowResult<V> {
    fn default() -> Self {
        Self {
            entry_states: BTreeMap::new(),
            exit_states: BTreeMap::new(),
            iterations: 0,
            converged: false,
        }
    }
}

// ---------------------------------------------------------------------------
// DataflowAnalysis
// ---------------------------------------------------------------------------

/// A configurable dataflow analysis driver.
///
/// Combines a [`ControlFlowGraph`], a lattice domain, and a transfer function
/// into a fixpoint computation.
#[derive(Debug, Clone)]
pub struct DataflowAnalysis<V: Lattice + Serialize> {
    /// Configuration for the underlying fixpoint engine.
    pub config: FixpointConfig,
    /// Initial abstract value (used for the entry block).
    pub initial_value: V,
    /// Whether to run the analysis in forward direction.
    pub forward: bool,
}

impl<V: Lattice + Serialize> DataflowAnalysis<V> {
    /// Create a new forward dataflow analysis with the given entry state.
    pub fn forward(initial_value: V) -> Self {
        Self {
            config: FixpointConfig::default(),
            initial_value,
            forward: true,
        }
    }

    /// Create a new backward dataflow analysis with the given exit state.
    pub fn backward(initial_value: V) -> Self {
        Self {
            config: FixpointConfig::default(),
            initial_value,
            forward: false,
        }
    }

    /// Override the fixpoint configuration.
    pub fn with_config(mut self, config: FixpointConfig) -> Self {
        self.config = config;
        self
    }

    /// Run the analysis on the given CFG and return the result.
    ///
    /// The `transfer` closure maps `(block_id, &entry_state)` to the
    /// exit state for that block.
    pub fn run<F>(&self, transfer: F) -> DataflowResult<V>
    where
        F: FnMut(u64, &V) -> V,
    {
        let mut engine = FixpointEngine::new(self.config.clone());
        // Seed the engine with the initial value at block 0 (entry).
        engine.set_initial(0, self.initial_value.clone());

        let fp: FixpointResult<V> = engine.solve(transfer, |_| vec![]);

        DataflowResult {
            entry_states: fp.solution.clone(),
            exit_states: fp.solution,
            iterations: fp.iterations,
            converged: fp.converged,
        }
    }
}
