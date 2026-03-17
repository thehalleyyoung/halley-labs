//! Conservation manifold definitions.
use serde::{Serialize, Deserialize};

/// A conservation manifold defined by constraint equations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationManifold { pub name: String, pub dimension: usize, pub codimension: usize }

impl ConservationManifold {
    pub fn new(name: impl Into<String>, dim: usize, codim: usize) -> Self {
        Self { name: name.into(), dimension: dim, codimension: codim }
    }
}
