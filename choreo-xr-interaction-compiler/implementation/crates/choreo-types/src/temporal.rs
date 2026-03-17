//! Temporal types.

use serde::{Deserialize, Serialize};

/// A point in time (seconds).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TimePoint(pub f64);

/// Duration in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Duration(pub f64);

/// A time interval [start, end).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: TimePoint,
    pub end: TimePoint,
}

/// Allen's interval algebra relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllenRelation {
    Before,
    Meets,
    Overlaps,
    Starts,
    During,
    Finishes,
    Equal,
    FinishedBy,
    Contains,
    StartedBy,
    OverlappedBy,
    MetBy,
    After,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub relation: AllenRelation,
    pub interval: Option<TimeInterval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MTLFormula {
    Atom(String),
    Not(Box<MTLFormula>),
    And(Box<MTLFormula>, Box<MTLFormula>),
    Or(Box<MTLFormula>, Box<MTLFormula>),
    Until(Box<MTLFormula>, Box<MTLFormula>, TimingBound),
    Eventually(Box<MTLFormula>, TimingBound),
    Always(Box<MTLFormula>, TimingBound),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedMTL {
    pub formula: MTLFormula,
    pub bound: TimingBound,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimingBound {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemporalPredicateId(pub String);
