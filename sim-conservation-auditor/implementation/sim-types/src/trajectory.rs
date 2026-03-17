use serde::{Deserialize, Serialize};
use crate::state::SimulationState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub states: Vec<SimulationState>,
}

impl Trajectory {
    pub fn new(states: Vec<SimulationState>) -> Self {
        Self { states }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn times(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.time).collect()
    }

    pub fn duration(&self) -> f64 {
        if self.states.len() < 2 {
            return 0.0;
        }
        self.states.last().unwrap().time - self.states.first().unwrap().time
    }

    pub fn state_at(&self, index: usize) -> Option<&SimulationState> {
        self.states.get(index)
    }
}
