//! Backward error analysis.
use serde::{Serialize, Deserialize};

/// Modified equation terms.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModifiedEquation { pub order: u32, pub terms: Vec<String> }

/// Modified Hamiltonian computation result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModifiedHamiltonian { pub original_energy: f64, pub modified_energy: f64, pub correction_terms: Vec<f64> }

/// Shadow orbit computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowOrbit { pub shadow_distance: f64, pub shadow_time: f64 }

/// Backward error analysis engine.
#[derive(Debug, Clone, Default)]
pub struct BackwardErrorAnalyzer;
impl BackwardErrorAnalyzer {
    /// Estimate the modified Hamiltonian for a numerical trajectory.
    pub fn modified_hamiltonian(&self, energies: &[f64]) -> ModifiedHamiltonian {
        let orig = energies.first().copied().unwrap_or(0.0);
        let last = energies.last().copied().unwrap_or(0.0);
        ModifiedHamiltonian { original_energy: orig, modified_energy: last, correction_terms: vec![last - orig] }
    }
}
