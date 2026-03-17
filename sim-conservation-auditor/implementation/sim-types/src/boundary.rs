use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryKind {
    Dirichlet,
    Neumann,
    Periodic,
    Reflecting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub kind: BoundaryKind,
    pub value: f64,
}

impl BoundaryCondition {
    pub fn new(kind: BoundaryKind, value: f64) -> Self {
        Self { kind, value }
    }
}
