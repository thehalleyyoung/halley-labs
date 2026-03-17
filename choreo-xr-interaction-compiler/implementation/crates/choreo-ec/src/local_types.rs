//! Local type equivalents for choreo-types and choreo-spatial.
//!
//! These types mirror the interfaces declared in the sibling crates.
//! When those crates gain implementations, this module can be replaced
//! with re-exports.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─── Geometry ────────────────────────────────────────────────────────────────

/// 3D point in world space.
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

    pub fn lerp(&self, other: &Point3, t: f64) -> Point3 {
        Point3 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl Eq for Point3 {}

impl std::hash::Hash for Point3 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        OrderedFloat(self.x).hash(state);
        OrderedFloat(self.y).hash(state);
        OrderedFloat(self.z).hash(state);
    }
}

/// 3D direction vector.
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

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let m = self.magnitude();
        if m < 1e-12 {
            return Self::zero();
        }
        Self::new(self.x / m, self.y / m, self.z / m)
    }

    pub fn dot(&self, other: &Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn lerp(&self, other: &Vector3, t: f64) -> Vector3 {
        Vector3 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

/// Quaternion for rotation representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn slerp(&self, other: &Quaternion, t: f64) -> Quaternion {
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let dot = dot.clamp(-1.0, 1.0);
        let theta = dot.abs().acos();
        if theta.abs() < 1e-6 {
            return *self;
        }
        let sin_theta = theta.sin();
        let w1 = ((1.0 - t) * theta).sin() / sin_theta;
        let w2 = (t * theta).sin() / sin_theta;
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        Quaternion {
            w: w1 * self.w + w2 * sign * other.w,
            x: w1 * self.x + w2 * sign * other.x,
            y: w1 * self.y + w2 * sign * other.y,
            z: w1 * self.z + w2 * sign * other.z,
        }
    }
}

/// Affine transform in 3D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform3D {
    pub position: Point3,
    pub rotation: Quaternion,
    pub scale: Vector3,
}

impl Transform3D {
    pub fn identity() -> Self {
        Self {
            position: Point3::origin(),
            rotation: Quaternion::identity(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn from_position(p: Point3) -> Self {
        Self {
            position: p,
            rotation: Quaternion::identity(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn lerp(&self, other: &Transform3D, t: f64) -> Transform3D {
        Transform3D {
            position: self.position.lerp(&other.position, t),
            rotation: self.rotation.slerp(&other.rotation, t),
            scale: self.scale.lerp(&other.scale, t),
        }
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
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    pub fn merged(&self, other: &AABB) -> AABB {
        AABB {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

// ─── Spatial ─────────────────────────────────────────────────────────────────

/// Unique identifier for a scene entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u64);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "entity_{}", self.0)
    }
}

/// Unique identifier for a spatial region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub u64);

/// Unique identifier for an interaction zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZoneId(pub u64);

/// Unique identifier for a spatial predicate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialPredicateId(pub u64);

impl fmt::Display for SpatialPredicateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sp_{}", self.0)
    }
}

/// Spatial predicate: a named relation over entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialPredicate {
    /// Entity A is within `distance` metres of entity B.
    Near { a: EntityId, b: EntityId, distance: f64 },
    /// Entity A is inside region R.
    Inside { entity: EntityId, region: RegionId },
    /// Entity A is touching entity B (surfaces overlap/contact).
    Touching { a: EntityId, b: EntityId },
    /// The AABBs of A and B overlap.
    Overlapping { a: EntityId, b: EntityId },
    /// Entity A is gazing at entity B (forward vector intersects B's bounds).
    GazingAt { gazer: EntityId, target: EntityId },
    /// Entity A is above entity B.
    Above { a: EntityId, b: EntityId },
    /// Entity A is facing entity B (dot product of forward and direction > threshold).
    Facing { a: EntityId, b: EntityId, threshold: f64 },
    /// Custom named predicate.
    Custom { name: String, entities: Vec<EntityId>, params: HashMap<String, f64> },
}

/// Boolean valuation of a spatial predicate at a moment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredicateValuation {
    pub predicate_id: SpatialPredicateId,
    pub predicate: SpatialPredicate,
    pub value: bool,
}

/// A region in space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialRegion {
    Sphere { center: Point3, radius: f64 },
    Box(AABB),
    Cylinder { center: Point3, radius: f64, half_height: f64 },
}

impl SpatialRegion {
    pub fn contains(&self, p: &Point3) -> bool {
        match self {
            SpatialRegion::Sphere { center, radius } => center.distance_to(p) <= *radius,
            SpatialRegion::Box(aabb) => aabb.contains_point(p),
            SpatialRegion::Cylinder { center, radius, half_height } => {
                let dx = p.x - center.x;
                let dz = p.z - center.z;
                let horiz = (dx * dx + dz * dz).sqrt();
                horiz <= *radius && (p.y - center.y).abs() <= *half_height
            }
        }
    }
}

/// An entity in the scene with transform and bounding volume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: EntityId,
    pub name: String,
    pub transform: Transform3D,
    pub bounds: AABB,
    pub forward: Vector3,
    pub properties: HashMap<String, String>,
}

impl SceneEntity {
    pub fn new(id: EntityId, name: impl Into<String>, transform: Transform3D) -> Self {
        Self {
            id,
            name: name.into(),
            transform,
            bounds: AABB::new(
                Point3::new(
                    transform.position.x - 0.5,
                    transform.position.y - 0.5,
                    transform.position.z - 0.5,
                ),
                Point3::new(
                    transform.position.x + 0.5,
                    transform.position.y + 0.5,
                    transform.position.z + 0.5,
                ),
            ),
            forward: Vector3::new(0.0, 0.0, 1.0),
            properties: HashMap::new(),
        }
    }
}

/// A complete scene configuration: all entities and regions at a point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneConfiguration {
    pub entities: HashMap<EntityId, SceneEntity>,
    pub regions: HashMap<RegionId, SpatialRegion>,
}

impl SceneConfiguration {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            regions: HashMap::new(),
        }
    }

    pub fn add_entity(&mut self, entity: SceneEntity) {
        self.entities.insert(entity.id, entity);
    }

    pub fn add_region(&mut self, id: RegionId, region: SpatialRegion) {
        self.regions.insert(id, region);
    }
}

impl Default for SceneConfiguration {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Temporal ────────────────────────────────────────────────────────────────

/// A point in time, represented as seconds from an epoch.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TimePoint(pub f64);

impl Eq for TimePoint {}

impl Ord for TimePoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

impl std::hash::Hash for TimePoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        OrderedFloat(self.0).hash(state);
    }
}

impl TimePoint {
    pub fn zero() -> Self {
        Self(0.0)
    }

    pub fn from_secs(s: f64) -> Self {
        Self(s)
    }

    pub fn as_secs(&self) -> f64 {
        self.0
    }

    pub fn advance(&self, dt: Duration) -> TimePoint {
        TimePoint(self.0 + dt.0)
    }
}

impl fmt::Display for TimePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}s", self.0)
    }
}

/// A duration in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Duration(pub f64);

impl Eq for Duration {}

impl Ord for Duration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

impl std::hash::Hash for Duration {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        OrderedFloat(self.0).hash(state);
    }
}

impl Duration {
    pub fn zero() -> Self {
        Self(0.0)
    }

    pub fn from_secs(s: f64) -> Self {
        Self(s)
    }

    pub fn from_millis(ms: f64) -> Self {
        Self(ms / 1000.0)
    }

    pub fn as_secs(&self) -> f64 {
        self.0
    }

    pub fn as_millis(&self) -> f64 {
        self.0 * 1000.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.abs() < 1e-12
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}s", self.0)
    }
}

impl std::ops::Add for Duration {
    type Output = Duration;
    fn add(self, rhs: Duration) -> Duration {
        Duration(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Duration {
    type Output = Duration;
    fn sub(self, rhs: Duration) -> Duration {
        Duration(self.0 - rhs.0)
    }
}

/// A time interval [start, end].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeInterval {
    pub fn new(start: TimePoint, end: TimePoint) -> Self {
        Self { start, end }
    }

    pub fn duration(&self) -> Duration {
        Duration(self.end.0 - self.start.0)
    }

    pub fn contains(&self, t: TimePoint) -> bool {
        t.0 >= self.start.0 && t.0 <= self.end.0
    }

    pub fn overlaps(&self, other: &TimeInterval) -> bool {
        self.start.0 <= other.end.0 && self.end.0 >= other.start.0
    }

    pub fn meets(&self, other: &TimeInterval) -> bool {
        (self.end.0 - other.start.0).abs() < 1e-9
    }
}

/// Allen's interval algebra relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllenRelation {
    Before,
    Meets,
    Overlaps,
    During,
    Starts,
    Finishes,
    Equal,
    After,
    MetBy,
    OverlappedBy,
    Contains,
    StartedBy,
    FinishedBy,
}

/// Temporal constraint between intervals or time points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalConstraint {
    Before { a: TimePoint, b: TimePoint },
    Within { point: TimePoint, interval: TimeInterval },
    MinDelay { from: TimePoint, to: TimePoint, min: Duration },
    MaxDelay { from: TimePoint, to: TimePoint, max: Duration },
    AllenConstraint { a: TimeInterval, b: TimeInterval, relation: AllenRelation },
    Deadline { point: TimePoint, deadline: TimePoint },
}

/// Unique identifier for a temporal predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemporalPredicateId(pub u64);

// ─── Events ──────────────────────────────────────────────────────────────────

/// Unique event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub u64);

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evt_{}", self.0)
    }
}

/// Which hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
}

/// Type of gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GestureType {
    Pinch,
    Grab,
    Poke,
    Point,
    Wave,
    Swipe,
    Tap,
    Custom(u32),
}

/// Type of action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    Activate,
    Deactivate,
    Toggle,
    Increment,
    Decrement,
    Reset,
    Custom(u32),
}

/// The kind of event that occurred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventKind {
    /// A hand gesture was recognized.
    Gesture { gesture: GestureType, hand: HandSide, entity: EntityId },
    /// An action was performed on an entity.
    Action { action: ActionType, entity: EntityId },
    /// A spatial predicate became true or false.
    SpatialChange { predicate_id: SpatialPredicateId, new_value: bool },
    /// A timer expired.
    TimerExpired { name: String },
    /// A timer was started.
    TimerStarted { name: String, duration: Duration },
    /// Gaze entered an entity's region.
    GazeEnter { entity: EntityId },
    /// Gaze left an entity's region.
    GazeExit { entity: EntityId },
    /// A collision started between two entities.
    CollisionStart { a: EntityId, b: EntityId },
    /// A collision ended between two entities.
    CollisionEnd { a: EntityId, b: EntityId },
    /// A custom named event.
    Custom { name: String, params: HashMap<String, String> },
    /// An internal system event.
    System { tag: String },
}

/// A timestamped event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub time: TimePoint,
    pub kind: EventKind,
}

impl Event {
    pub fn new(id: EventId, time: TimePoint, kind: EventKind) -> Self {
        Self { id, time, kind }
    }
}

/// A pattern that matches events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventPattern {
    /// Match any event.
    Any,
    /// Match a specific event kind.
    KindMatch(EventKind),
    /// Match by gesture type regardless of hand/entity.
    GestureMatch(GestureType),
    /// Match by action type.
    ActionMatch(ActionType),
    /// Match spatial change for a specific predicate.
    SpatialChangeMatch(SpatialPredicateId),
    /// Match by custom event name.
    NamedMatch(String),
    /// Match if any sub-pattern matches.
    AnyOf(Vec<EventPattern>),
    /// Match if all sub-patterns match.
    AllOf(Vec<EventPattern>),
}

impl EventPattern {
    pub fn matches(&self, event: &EventKind) -> bool {
        match self {
            EventPattern::Any => true,
            EventPattern::KindMatch(k) => k == event,
            EventPattern::GestureMatch(g) => {
                matches!(event, EventKind::Gesture { gesture, .. } if gesture == g)
            }
            EventPattern::ActionMatch(a) => {
                matches!(event, EventKind::Action { action, .. } if action == a)
            }
            EventPattern::SpatialChangeMatch(id) => {
                matches!(event, EventKind::SpatialChange { predicate_id, .. } if predicate_id == id)
            }
            EventPattern::NamedMatch(n) => {
                matches!(event, EventKind::Custom { name, .. } if name == n)
            }
            EventPattern::AnyOf(ps) => ps.iter().any(|p| p.matches(event)),
            EventPattern::AllOf(ps) => ps.iter().all(|p| p.matches(event)),
        }
    }
}

/// An ordered sequence of events.
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

    pub fn sorted(&self) -> Self {
        let mut events = self.events.clone();
        events.sort_by(|a, b| a.time.cmp(&b.time));
        Self { events }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn time_span(&self) -> Option<TimeInterval> {
        if self.events.is_empty() {
            return None;
        }
        let sorted = self.sorted();
        Some(TimeInterval::new(
            sorted.events.first().unwrap().time,
            sorted.events.last().unwrap().time,
        ))
    }

    pub fn events_in(&self, interval: &TimeInterval) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| interval.contains(e.time))
            .collect()
    }
}

impl Default for EventTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Automaton ───────────────────────────────────────────────────────────────

/// Unique state identifier within an automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateId(pub u64);

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s_{}", self.0)
    }
}

/// Unique transition identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionId(pub u64);

impl fmt::Display for TransitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t_{}", self.0)
    }
}

/// Timer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimerId(pub u64);

/// Variable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub u64);

/// Dynamically-typed value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Entity(EntityId),
    Null,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(v) => write!(f, "{}", v),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Entity(e) => write!(f, "{}", e),
            Value::Null => write!(f, "null"),
        }
    }
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(v.0),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Guard condition on an automaton transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Guard {
    EventMatch(EventPattern),
    FluentHolds(String),
    FluentNotHolds(String),
    TimerExpired(String),
    And(Vec<Guard>),
    Or(Vec<Guard>),
    Not(Box<Guard>),
    True,
}

/// Action to execute when a transition fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    SetFluent { name: String, value: Value },
    StartTimer { name: String, duration: Duration },
    StopTimer { name: String },
    EmitEvent(EventKind),
    Custom { name: String, params: HashMap<String, String> },
}

/// Automaton transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
    pub actions: Vec<Action>,
    pub priority: i32,
}

/// State in an automaton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub invariants: Vec<Guard>,
}

/// Kind of automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutomatonKind {
    Interaction,
    Gesture,
    Spatial,
    Timer,
    Composite,
}

/// Complete automaton definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutomatonDef {
    pub name: String,
    pub kind: AutomatonKind,
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,
    pub initial_state: StateId,
    pub accepting_states: Vec<StateId>,
}

/// Product state of composed automata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductState(pub Vec<StateId>);

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors specific to the EC engine.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ECError {
    #[error("fluent not found: {0}")]
    FluentNotFound(String),

    #[error("axiom evaluation failed: {0}")]
    AxiomEvaluation(String),

    #[error("spatial oracle error: {0}")]
    SpatialOracleError(String),

    #[error("narrative violation: {0}")]
    NarrativeViolation(String),

    #[error("compilation error: {0}")]
    CompilationError(String),

    #[error("domain error: {0}")]
    DomainError(String),

    #[error("time out of bounds: {0}")]
    TimeOutOfBounds(String),

    #[error("circular dependency detected: {0}")]
    CircularDependency(String),

    #[error("conflicting axioms: {0}")]
    ConflictingAxioms(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("trace comparison error: {0}")]
    TraceError(String),

    #[error("serialization error: {0}")]
    SerializationError(String),
}

pub type ECResult<T> = Result<T, ECError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_point3_lerp() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 20.0, 30.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-9);
        assert!((mid.y - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_aabb_intersection() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        let b = AABB::new(Point3::new(1.0, 1.0, 1.0), Point3::new(3.0, 3.0, 3.0));
        assert!(a.intersects(&b));

        let c = AABB::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_time_point_ordering() {
        let a = TimePoint::from_secs(1.0);
        let b = TimePoint::from_secs(2.0);
        assert!(a < b);
        assert_eq!(a.advance(Duration::from_secs(1.0)), b);
    }

    #[test]
    fn test_time_interval_contains() {
        let interval = TimeInterval::new(
            TimePoint::from_secs(1.0),
            TimePoint::from_secs(5.0),
        );
        assert!(interval.contains(TimePoint::from_secs(3.0)));
        assert!(!interval.contains(TimePoint::from_secs(6.0)));
    }

    #[test]
    fn test_event_pattern_matching() {
        let pattern = EventPattern::GestureMatch(GestureType::Grab);
        let event = EventKind::Gesture {
            gesture: GestureType::Grab,
            hand: HandSide::Right,
            entity: EntityId(1),
        };
        assert!(pattern.matches(&event));

        let other = EventKind::Action {
            action: ActionType::Activate,
            entity: EntityId(1),
        };
        assert!(!pattern.matches(&other));
    }

    #[test]
    fn test_event_trace_sorting() {
        let mut trace = EventTrace::new();
        trace.push(Event::new(EventId(2), TimePoint::from_secs(3.0),
            EventKind::System { tag: "b".into() }));
        trace.push(Event::new(EventId(1), TimePoint::from_secs(1.0),
            EventKind::System { tag: "a".into() }));
        let sorted = trace.sorted();
        assert_eq!(sorted.events[0].id, EventId(1));
        assert_eq!(sorted.events[1].id, EventId(2));
    }

    #[test]
    fn test_spatial_region_contains() {
        let sphere = SpatialRegion::Sphere {
            center: Point3::origin(),
            radius: 5.0,
        };
        assert!(sphere.contains(&Point3::new(1.0, 1.0, 1.0)));
        assert!(!sphere.contains(&Point3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_value_conversions() {
        let v = Value::Float(OrderedFloat(3.14));
        assert!((v.as_f64().unwrap() - 3.14).abs() < 1e-9);
        assert!(v.as_bool().is_none());

        let v = Value::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_scene_configuration() {
        let mut scene = SceneConfiguration::new();
        scene.add_entity(SceneEntity::new(
            EntityId(1),
            "test_entity",
            Transform3D::from_position(Point3::new(1.0, 2.0, 3.0)),
        ));
        assert_eq!(scene.entities.len(), 1);
        assert!(scene.entities.contains_key(&EntityId(1)));
    }
}
