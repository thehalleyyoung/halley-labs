use serde::{Deserialize, Serialize};
use crate::Vec3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseSpacePoint {
    pub q: Vec3,
    pub p: Vec3,
}

impl PhaseSpacePoint {
    pub fn new(q: Vec3, p: Vec3) -> Self {
        Self { q, p }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticForm {
    pub dimension: usize,
}

impl SymplecticForm {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}
