//! Event types.

use serde::{Deserialize, Serialize};
use crate::temporal::TimePoint;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    GestureStart(GestureType),
    GestureEnd(GestureType),
    Action(ActionType),
    SpatialEnter,
    SpatialExit,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub kind: EventKind,
    pub timestamp: TimePoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTrace {
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPattern {
    Single(EventKind),
    Sequence(Vec<EventPattern>),
    Choice(Vec<EventPattern>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStream {
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    Grab,
    Release,
    Point,
    Press,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GestureType {
    Pinch,
    Swipe,
    Tap,
    Wave,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
    Both,
}
