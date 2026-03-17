//! State machine compilation, spatial event automata, product composition,
//! and automata optimization for the Choreo XR Interaction Compiler.
//!
//! This crate provides:
//! - **builder**: Fluent API for constructing automata from interaction specifications
//! - **automaton**: Core spatial event automaton with guard evaluation and transition firing
//! - **product**: Product automaton composition with on-the-fly and symbolic methods
//! - **optimization**: Multi-pass optimization pipeline for automata
//! - **nfa**: NFA-specific operations including Thompson construction and subset conversion
//! - **bdd**: Binary Decision Diagrams for symbolic state representation
//! - **analysis**: Structural analysis, SCC detection, bisimulation, and language operations

pub mod automaton;
pub mod builder;
pub mod product;
pub mod optimization;
pub mod nfa;
pub mod bdd;
pub mod analysis;

// ---------------------------------------------------------------------------
// Local equivalents for types that would normally come from dependency crates.
// These mirror the interfaces declared in choreo-types, choreo-spatial, and
// choreo-ec so that choreo-automata is self-contained.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---- Identifiers ----------------------------------------------------------

/// Unique state identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct StateId(pub u32);

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Unique transition identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TransitionId(pub u32);

impl fmt::Display for TransitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Unique entity identifier (objects, hands, controllers in a scene).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Unique region identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub String);

/// Unique zone identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZoneId(pub String);

/// Unique event identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

/// Unique spatial predicate identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SpatialPredicateId(pub String);

/// Unique temporal predicate identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct TemporalPredicateId(pub String);

/// Timer identifier used in temporal guards.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimerId(pub String);

/// Variable identifier for automaton data variables.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub String);

// ---- Geometry (minimal) ---------------------------------------------------

/// 3-D point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// 3-D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    pub min: Point3,
    pub max: Point3,
}

impl AABB {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }
    pub fn contains_point(&self, p: &Point3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// Bounding sphere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
}

// ---- Temporal primitives --------------------------------------------------

/// Point in time (seconds since epoch / scene start).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TimePoint(pub f64);

impl TimePoint {
    pub fn zero() -> Self {
        Self(0.0)
    }
}

/// Duration in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Duration(pub f64);

/// Closed time interval `[start, end]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeInterval {
    pub fn new(start: TimePoint, end: TimePoint) -> Self {
        Self { start, end }
    }
    pub fn contains(&self, t: TimePoint) -> bool {
        t.0 >= self.start.0 && t.0 <= self.end.0
    }
    pub fn duration(&self) -> Duration {
        Duration(self.end.0 - self.start.0)
    }
}

// ---- Spatial types --------------------------------------------------------

/// Spatial predicate – a boolean condition over the spatial scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialPredicate {
    Inside { entity: EntityId, region: RegionId },
    Proximity {
        entity_a: EntityId,
        entity_b: EntityId,
        threshold: f64,
    },
    GazeAt { entity: EntityId, target: RegionId },
    Contact {
        entity_a: EntityId,
        entity_b: EntityId,
    },
    Grasping {
        hand: EntityId,
        object: EntityId,
    },
    Not(Box<SpatialPredicate>),
    And(Vec<SpatialPredicate>),
    Or(Vec<SpatialPredicate>),
    Named(SpatialPredicateId),
}

/// Spatial region – an abstract or concrete volume in 3-D space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialRegion {
    Sphere(Sphere),
    AABB(AABB),
    Named(RegionId),
}

/// Scene entity – an object present in the spatial scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: EntityId,
    pub position: Point3,
    pub bounding: Option<AABB>,
    pub properties: HashMap<String, String>,
}

/// Scene configuration – a snapshot of the spatial world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneConfiguration {
    pub entities: Vec<SceneEntity>,
    pub regions: HashMap<RegionId, SpatialRegion>,
    pub timestamp: TimePoint,
}

impl SceneConfiguration {
    pub fn empty() -> Self {
        Self {
            entities: Vec::new(),
            regions: HashMap::new(),
            timestamp: TimePoint::zero(),
        }
    }
    pub fn entity(&self, id: &EntityId) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| &e.id == id)
    }
}

// ---- Events ---------------------------------------------------------------

/// Kinds of events recognised by the interaction compiler.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    GazeEnter,
    GazeExit,
    GazeDwell,
    GrabStart,
    GrabEnd,
    TouchStart,
    TouchEnd,
    ProximityEnter,
    ProximityExit,
    GestureRecognised(String),
    ButtonPress(String),
    ButtonRelease(String),
    Timer(TimerId),
    Custom(String),
    Epsilon,
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventKind::GazeEnter => write!(f, "gaze_enter"),
            EventKind::GazeExit => write!(f, "gaze_exit"),
            EventKind::GazeDwell => write!(f, "gaze_dwell"),
            EventKind::GrabStart => write!(f, "grab_start"),
            EventKind::GrabEnd => write!(f, "grab_end"),
            EventKind::TouchStart => write!(f, "touch_start"),
            EventKind::TouchEnd => write!(f, "touch_end"),
            EventKind::ProximityEnter => write!(f, "proximity_enter"),
            EventKind::ProximityExit => write!(f, "proximity_exit"),
            EventKind::GestureRecognised(g) => write!(f, "gesture({})", g),
            EventKind::ButtonPress(b) => write!(f, "press({})", b),
            EventKind::ButtonRelease(b) => write!(f, "release({})", b),
            EventKind::Timer(t) => write!(f, "timer({})", t.0),
            EventKind::Custom(c) => write!(f, "custom({})", c),
            EventKind::Epsilon => write!(f, "ε"),
        }
    }
}

/// A concrete event occurrence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub kind: EventKind,
    pub timestamp: TimePoint,
    pub source: Option<EntityId>,
    pub target: Option<EntityId>,
    pub data: HashMap<String, String>,
}

/// An ordered trace of events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventTrace {
    pub events: Vec<Event>,
}

impl EventTrace {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }
    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }
    pub fn len(&self) -> usize {
        self.events.len()
    }
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for EventTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Guards & Actions -----------------------------------------------------

/// Guard on a transition – evaluated against the current scene & time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Guard {
    True,
    False,
    Spatial(SpatialPredicate),
    Temporal(TemporalGuardExpr),
    Event(EventKind),
    And(Vec<Guard>),
    Or(Vec<Guard>),
    Not(Box<Guard>),
}

impl Guard {
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, Guard::True)
    }
    pub fn is_trivially_false(&self) -> bool {
        matches!(self, Guard::False)
    }
    pub fn and(self, other: Guard) -> Guard {
        match (&self, &other) {
            (Guard::True, _) => other,
            (_, Guard::True) => self,
            (Guard::False, _) | (_, Guard::False) => Guard::False,
            _ => Guard::And(vec![self, other]),
        }
    }
    pub fn or(self, other: Guard) -> Guard {
        match (&self, &other) {
            (Guard::True, _) | (_, Guard::True) => Guard::True,
            (Guard::False, _) => other,
            (_, Guard::False) => self,
            _ => Guard::Or(vec![self, other]),
        }
    }
    pub fn not(self) -> Guard {
        match self {
            Guard::True => Guard::False,
            Guard::False => Guard::True,
            Guard::Not(inner) => *inner,
            other => Guard::Not(Box::new(other)),
        }
    }

    /// Collect referenced spatial predicate ids.
    pub fn spatial_predicate_ids(&self) -> Vec<SpatialPredicateId> {
        let mut ids = Vec::new();
        self.collect_spatial_ids(&mut ids);
        ids
    }

    fn collect_spatial_ids(&self, ids: &mut Vec<SpatialPredicateId>) {
        match self {
            Guard::Spatial(SpatialPredicate::Named(id)) => ids.push(id.clone()),
            Guard::Spatial(SpatialPredicate::And(preds)) => {
                for p in preds {
                    Guard::Spatial(p.clone()).collect_spatial_ids(ids);
                }
            }
            Guard::Spatial(SpatialPredicate::Or(preds)) => {
                for p in preds {
                    Guard::Spatial(p.clone()).collect_spatial_ids(ids);
                }
            }
            Guard::Spatial(SpatialPredicate::Not(inner)) => {
                Guard::Spatial((**inner).clone()).collect_spatial_ids(ids);
            }
            Guard::And(gs) | Guard::Or(gs) => {
                for g in gs {
                    g.collect_spatial_ids(ids);
                }
            }
            Guard::Not(g) => g.collect_spatial_ids(ids),
            _ => {}
        }
    }

    /// Count the number of leaf predicates in this guard.
    pub fn complexity(&self) -> usize {
        match self {
            Guard::True | Guard::False => 0,
            Guard::Spatial(_) | Guard::Temporal(_) | Guard::Event(_) => 1,
            Guard::And(gs) | Guard::Or(gs) => gs.iter().map(|g| g.complexity()).sum::<usize>() + 1,
            Guard::Not(g) => g.complexity() + 1,
        }
    }
}

impl fmt::Display for Guard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Guard::True => write!(f, "true"),
            Guard::False => write!(f, "false"),
            Guard::Spatial(sp) => write!(f, "spatial({:?})", sp),
            Guard::Temporal(tg) => write!(f, "temporal({:?})", tg),
            Guard::Event(ek) => write!(f, "event({})", ek),
            Guard::And(gs) => {
                let parts: Vec<String> = gs.iter().map(|g| format!("{}", g)).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            Guard::Or(gs) => {
                let parts: Vec<String> = gs.iter().map(|g| format!("{}", g)).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            Guard::Not(g) => write!(f, "¬{}", g),
        }
    }
}

/// Temporal guard expression – condition on timers and durations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalGuardExpr {
    TimerElapsed { timer: TimerId, threshold: Duration },
    WithinInterval(TimeInterval),
    Named(TemporalPredicateId),
    And(Vec<TemporalGuardExpr>),
    Or(Vec<TemporalGuardExpr>),
    Not(Box<TemporalGuardExpr>),
}

/// Actions produced when a transition fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    StartTimer(TimerId),
    StopTimer(TimerId),
    EmitEvent(EventKind),
    SetVar { var: VarId, value: Value },
    PlayFeedback(String),
    Highlight { entity: EntityId, style: String },
    ClearHighlight(EntityId),
    MoveEntity { entity: EntityId, target: Point3 },
    Custom(String),
    Noop,
}

/// Run-time value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

// ---- EC Blueprint (from choreo-ec) ----------------------------------------

/// A blueprint produced by the Event Calculus compiler that the automaton
/// builder can convert into a `SpatialEventAutomaton`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECBlueprint {
    pub name: String,
    pub fluents: Vec<ECFluent>,
    pub event_rules: Vec<ECEventRule>,
    pub initial_fluents: Vec<(String, bool)>,
}

/// A fluent in the EC model (a boolean proposition that changes over time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECFluent {
    pub name: String,
    pub predicate: Option<SpatialPredicate>,
}

/// A rule describing how events change fluents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECEventRule {
    pub trigger_event: EventKind,
    pub guard: Guard,
    pub initiates: Vec<String>,
    pub terminates: Vec<String>,
}

// ---- Source span ----------------------------------------------------------

/// Byte-offset span in source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
    pub fn empty() -> Self {
        Self { start: 0, end: 0 }
    }
}

// ---- Errors ---------------------------------------------------------------

/// Errors produced during automaton construction and analysis.
#[derive(Debug, thiserror::Error)]
pub enum AutomataError {
    #[error("No initial state set")]
    NoInitialState,
    #[error("Duplicate initial state: {0}")]
    DuplicateInitialState(StateId),
    #[error("State {0} not found")]
    StateNotFound(StateId),
    #[error("Transition {0} not found")]
    TransitionNotFound(TransitionId),
    #[error("Unreachable states detected: {0:?}")]
    UnreachableStates(Vec<StateId>),
    #[error("Orphaned state: {0}")]
    OrphanedState(StateId),
    #[error("Guard consistency error: {0}")]
    GuardInconsistency(String),
    #[error("Invalid transition from {from} to {to}: {reason}")]
    InvalidTransition {
        from: StateId,
        to: StateId,
        reason: String,
    },
    #[error("Product composition error: {0}")]
    CompositionError(String),
    #[error("BDD error: {0}")]
    BddError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Blueprint conversion error: {0}")]
    BlueprintConversionError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AutomataError>;
