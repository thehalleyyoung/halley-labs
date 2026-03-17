//! Buggy scenarios for testing.
use serde::{Serialize, Deserialize};

/// Types of intentional bugs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioKind {
    EnergyLeak,
    MomentumDrift,
    SymplecticViolation,
    AngularMomentumError,
    NumericalInstability,
}

/// A buggy scenario for testing the conservation auditor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuggyScenario {
    pub name: String,
    pub kind: ScenarioKind,
    pub description: String,
    pub expected_violation: String,
}

impl BuggyScenario {
    /// Create a predefined set of buggy scenarios.
    pub fn all() -> Vec<Self> {
        vec![
            Self { name: "leaky_euler".into(), kind: ScenarioKind::EnergyLeak, description: "Forward Euler on Kepler problem".into(), expected_violation: "Energy increases monotonically".into() },
            Self { name: "momentum_asymmetry".into(), kind: ScenarioKind::MomentumDrift, description: "Asymmetric force computation".into(), expected_violation: "Linear momentum drifts".into() },
            Self { name: "broken_verlet".into(), kind: ScenarioKind::SymplecticViolation, description: "Verlet with incorrect velocity update".into(), expected_violation: "Phase space volume not preserved".into() },
        ]
    }
}
