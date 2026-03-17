//! Source-level mutation replay from bytecode descriptors.

use shared_types::{Expression, MutantDescriptor, MutationSite, Statement};

/// Replays a mutation at the source level.
pub struct MutationReplay;

impl MutationReplay {
    pub fn new() -> Self {
        Self
    }

    /// Replay a mutation described by a descriptor, returning the mutated statement.
    pub fn replay(&self, _descriptor: &MutantDescriptor) -> Option<Statement> {
        None
    }
}
