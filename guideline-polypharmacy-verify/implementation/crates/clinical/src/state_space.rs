//! Formal clinical state space, transitions, and predicates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A value in the clinical state space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateValue {
    Boolean(bool),
    Numeric(f64),
    Categorical(String),
    Absent,
}

/// A named variable in the state space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StateVariable(pub String);

impl StateVariable {
    pub fn new(name: &str) -> Self { StateVariable(name.to_string()) }
}

/// A snapshot of the full clinical state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClinicalState {
    pub variables: HashMap<String, StateValue>,
    pub timestamp_hours: f64,
}

impl ClinicalState {
    pub fn new() -> Self { ClinicalState { variables: HashMap::new(), timestamp_hours: 0.0 } }

    pub fn set(&mut self, var: &str, val: StateValue) {
        self.variables.insert(var.to_string(), val);
    }

    pub fn get(&self, var: &str) -> Option<&StateValue> {
        self.variables.get(var)
    }
}

/// A clinical action that causes a state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalAction {
    AdministerDose { drug: String, dose_mg: f64 },
    WithholdDose { drug: String, reason: String },
    OrderLab { name: String },
    AdjustDose { drug: String, new_dose_mg: f64 },
    Discontinue { drug: String },
    Consult { specialty: String },
}

/// A predicate over the clinical state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalPredicate {
    GreaterThan { variable: String, threshold: f64 },
    LessThan { variable: String, threshold: f64 },
    Equals { variable: String, value: StateValue },
    And(Vec<ClinicalPredicate>),
    Or(Vec<ClinicalPredicate>),
    Not(Box<ClinicalPredicate>),
}

impl ClinicalPredicate {
    pub fn evaluate(&self, state: &ClinicalState) -> bool {
        match self {
            ClinicalPredicate::GreaterThan { variable, threshold } => {
                matches!(state.get(variable), Some(StateValue::Numeric(v)) if *v > *threshold)
            }
            ClinicalPredicate::LessThan { variable, threshold } => {
                matches!(state.get(variable), Some(StateValue::Numeric(v)) if *v < *threshold)
            }
            ClinicalPredicate::Equals { variable, value } => {
                state.get(variable) == Some(value)
            }
            ClinicalPredicate::And(preds) => preds.iter().all(|p| p.evaluate(state)),
            ClinicalPredicate::Or(preds) => preds.iter().any(|p| p.evaluate(state)),
            ClinicalPredicate::Not(pred) => !pred.evaluate(state),
        }
    }
}

/// A transition between clinical states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub action: ClinicalAction,
    pub guard: Option<ClinicalPredicate>,
}

/// The clinical state space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClinicalStateSpace {
    pub states: Vec<ClinicalState>,
    pub transitions: Vec<StateTransition>,
}

/// History of states for a patient.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateHistory {
    pub snapshots: Vec<ClinicalState>,
}

impl StateHistory {
    pub fn new() -> Self { StateHistory { snapshots: Vec::new() } }

    pub fn push(&mut self, state: ClinicalState) { self.snapshots.push(state); }

    pub fn latest(&self) -> Option<&ClinicalState> { self.snapshots.last() }
}
