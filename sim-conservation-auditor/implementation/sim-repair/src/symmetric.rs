//! Symmetric (time-reversible) repair methods.

/// Applies symmetric corrections to maintain time-reversibility.
#[derive(Debug, Clone, Default)]
pub struct SymmetricRepair;
impl SymmetricRepair {
    /// Symmetrize a state update to preserve time-reversibility.
    pub fn symmetrize(state_forward: &[f64], state_backward: &[f64]) -> Vec<f64> {
        state_forward.iter().zip(state_backward).map(|(f, b)| 0.5 * (f + b)).collect()
    }
}
