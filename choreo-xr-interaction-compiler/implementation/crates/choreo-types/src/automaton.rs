//! Automaton types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimerId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(i64), // stored as bits for Eq/Hash compatibility
    String(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Guard {
    Predicate(String),
    And(Vec<Guard>),
    Or(Vec<Guard>),
    Not(Box<Guard>),
    True,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Emit(String),
    Assign(VarId, Value),
    StartTimer(TimerId),
    StopTimer(TimerId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub name: String,
    pub entry_actions: Vec<Action>,
    pub exit_actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomatonKind {
    Mealy,
    Moore,
    Timed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatonDef {
    pub kind: AutomatonKind,
    pub states: HashMap<StateId, State>,
    pub transitions: Vec<Transition>,
    pub initial: StateId,
    pub accepting: Vec<StateId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductState(pub Vec<StateId>);
