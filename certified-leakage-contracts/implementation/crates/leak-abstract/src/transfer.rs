//! Transfer functions that lift concrete instruction semantics into abstract
//! domains.
//!
//! A transfer function maps a pre-state and an instruction to a post-state in
//! the chosen abstract domain.  Forward and backward variants are provided to
//! support both forward dataflow problems (e.g., reachable cache states) and
//! backward problems (e.g., live variable analysis).

use shared_types::{Instruction, BlockId};

use crate::lattice::Lattice;

/// The most general transfer-function trait.
///
/// Takes a program point (block + instruction) and transforms an abstract
/// state `V` accordingly.
pub trait TransferFunction<V: Lattice> {
    /// Apply the transfer function to `state` at the given instruction,
    /// returning the updated abstract value.
    fn apply(&self, state: &V, instruction: &Instruction) -> V;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "transfer"
    }
}

/// Marker trait for *forward* transfer functions (pre-state → post-state).
///
/// Used by forward dataflow analyses such as cache state propagation.
pub trait ForwardTransfer<V: Lattice>: TransferFunction<V> {
    /// Transfer across an entire basic block in forward order.
    fn transfer_block(&self, state: &V, instructions: &[Instruction]) -> V {
        instructions.iter().fold(state.clone(), |acc, insn| self.apply(&acc, insn))
    }

    /// Transfer along a CFG edge (e.g., conditional branch filtering).
    fn transfer_edge(&self, state: &V, _from: BlockId, _to: BlockId) -> V {
        state.clone()
    }
}

/// Marker trait for *backward* transfer functions (post-state → pre-state).
///
/// Used by backward dataflow analyses such as liveness or demand-driven
/// observation queries.
pub trait BackwardTransfer<V: Lattice>: TransferFunction<V> {
    /// Transfer across an entire basic block in reverse order.
    fn transfer_block_backward(&self, state: &V, instructions: &[Instruction]) -> V {
        instructions.iter().rev().fold(state.clone(), |acc, insn| self.apply(&acc, insn))
    }

    /// Transfer along a CFG edge in the reverse direction.
    fn transfer_edge_backward(&self, state: &V, _from: BlockId, _to: BlockId) -> V {
        state.clone()
    }
}
