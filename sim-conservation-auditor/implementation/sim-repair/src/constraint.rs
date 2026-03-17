//! Constraint definitions for conservation repair.
use serde::{Serialize, Deserialize};

/// A conservation constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationConstraint { pub name: String, pub target_value: f64, pub tolerance: f64 }
impl ConservationConstraint {
    pub fn new(name: impl Into<String>, target: f64, tol: f64) -> Self {
        Self { name: name.into(), target_value: target, tolerance: tol }
    }
    pub fn is_satisfied(&self, current_value: f64) -> bool {
        (current_value - self.target_value).abs() < self.tolerance
    }
}
