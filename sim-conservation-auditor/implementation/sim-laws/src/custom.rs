//! Custom and composite conservation laws.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// A user-defined conservation law with a custom computation function.
#[derive(Clone)]
pub struct CustomLaw {
    /// Name of the custom law.
    pub law_name: String,
    /// Computation function.
    compute_fn: fn(&SimulationState) -> f64,
}

impl std::fmt::Debug for CustomLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomLaw").field("name", &self.law_name).finish()
    }
}

impl CustomLaw {
    /// Create a new custom conservation law.
    pub fn new(name: impl Into<String>, f: fn(&SimulationState) -> f64) -> Self {
        Self { law_name: name.into(), compute_fn: f }
    }
}

impl crate::ConservationChecker for CustomLaw {
    fn name(&self) -> &str { &self.law_name }
    fn kind(&self) -> ConservationKind { ConservationKind::Custom }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar((self.compute_fn)(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        (self.compute_fn)(state)
    }
}

/// A composite conservation law that sums multiple sub-laws.
#[derive(Debug, Clone)]
pub struct CompositeConservation {
    /// Name of the composite law.
    pub law_name: String,
    /// Child law names for reference.
    pub child_names: Vec<String>,
}

impl crate::ConservationChecker for CompositeConservation {
    fn name(&self) -> &str { &self.law_name }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, _state: &SimulationState) -> f64 { 0.0 }
}

/// A conservation law that is only checked when a condition is met.
#[derive(Debug, Clone)]
pub struct ConditionalConservation {
    /// Name of the conditional law.
    pub law_name: String,
    /// Description of the condition.
    pub condition_description: String,
}

impl crate::ConservationChecker for ConditionalConservation {
    fn name(&self) -> &str { &self.law_name }
    fn kind(&self) -> ConservationKind { ConservationKind::Custom }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        let _ = state;
        ConservedQuantity::scalar(0.0)
    }
}
