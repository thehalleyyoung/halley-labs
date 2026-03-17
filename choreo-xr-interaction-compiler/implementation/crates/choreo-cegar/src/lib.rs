//! Spatial CEGAR (Counterexample-Guided Abstraction Refinement) verification engine
//! with geometric refinement for the Choreo XR interaction compiler.
//!
//! This crate implements:
//! - Geometric abstraction of spatial interaction scenes
//! - CEGAR loop for iterative abstraction refinement
//! - Abstract model checking (explicit, symbolic, BMC)
//! - Counterexample analysis with GJK/EPA feasibility checking
//! - Geometric consistency pruning (monotonicity, triangle inequality, containment)
//! - Compositional verification via spatial separability
//! - Verification property specification and certificates

pub mod abstraction;
pub mod adaptive_decomposition;
pub mod cegar_loop;
pub mod certificate;
pub mod compositional;
pub mod counterexample;
pub mod model_checker;
pub mod properties;
pub mod pruning;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Local equivalents of types from choreo-types / choreo-spatial / choreo-automata
// ---------------------------------------------------------------------------

/// Unique identifier for an automaton state.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct StateId(pub u64);

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Unique identifier for a transition.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct TransitionId(pub u64);

impl fmt::Display for TransitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Entity identifier in the scene.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct EntityId(pub u64);

/// Region identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct RegionId(pub u64);

/// Zone identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct ZoneId(pub u64);

/// Spatial predicate identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct SpatialPredicateId(pub u64);

// ---------------------------------------------------------------------------
// Geometry primitives
// ---------------------------------------------------------------------------

/// A point in 3-D Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Point3 {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn midpoint(&self, other: &Point3) -> Point3 {
        Point3 {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
            z: (self.z + other.z) * 0.5,
        }
    }

    pub fn to_vec3(&self) -> Vector3 {
        Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
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

/// A 3-D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Vector3 {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
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

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Vector3 {
        let len = self.length();
        if len < 1e-12 {
            return Vector3::default();
        }
        Vector3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn scale(&self, s: f64) -> Vector3 {
        Vector3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn add(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn sub(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn negate(&self) -> Vector3 {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    pub min: Point3,
    pub max: Point3,
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(1.0, 1.0, 1.0),
        }
    }
}

impl AABB {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Point3 {
        self.min.midpoint(&self.max)
    }

    pub fn extents(&self) -> Vector3 {
        Vector3 {
            x: self.max.x - self.min.x,
            y: self.max.y - self.min.y,
            z: self.max.z - self.min.z,
        }
    }

    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    pub fn longest_axis(&self) -> usize {
        let e = self.extents();
        if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        }
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

    pub fn contains_aabb(&self, other: &AABB) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
            && self.min.z <= other.min.z
            && self.max.z >= other.max.z
    }

    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: Point3 {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
                z: self.min.z.min(other.min.z),
            },
            max: Point3 {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
                z: self.max.z.max(other.max.z),
            },
        }
    }

    pub fn intersection(&self, other: &AABB) -> Option<AABB> {
        let min = Point3 {
            x: self.min.x.max(other.min.x),
            y: self.min.y.max(other.min.y),
            z: self.min.z.max(other.min.z),
        };
        let max = Point3 {
            x: self.max.x.min(other.max.x),
            y: self.max.y.min(other.max.y),
            z: self.max.z.min(other.max.z),
        };
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(AABB { min, max })
        } else {
            None
        }
    }

    pub fn split_at_axis(&self, axis: usize, value: f64) -> (AABB, AABB) {
        let mut left = *self;
        let mut right = *self;
        match axis {
            0 => {
                left.max.x = value;
                right.min.x = value;
            }
            1 => {
                left.max.y = value;
                right.min.y = value;
            }
            _ => {
                left.max.z = value;
                right.min.z = value;
            }
        }
        (left, right)
    }

    pub fn axis_length(&self, axis: usize) -> f64 {
        match axis {
            0 => self.max.x - self.min.x,
            1 => self.max.y - self.min.y,
            _ => self.max.z - self.min.z,
        }
    }

    pub fn axis_center(&self, axis: usize) -> f64 {
        match axis {
            0 => (self.min.x + self.max.x) * 0.5,
            1 => (self.min.y + self.max.y) * 0.5,
            _ => (self.min.z + self.max.z) * 0.5,
        }
    }
}

/// A plane in 3-D defined by normal · p = distance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    pub normal: Vector3,
    pub distance: f64,
}

impl Plane {
    pub fn new(normal: Vector3, distance: f64) -> Self {
        Self { normal, distance }
    }

    pub fn signed_distance(&self, p: &Point3) -> f64 {
        self.normal.dot(&p.to_vec3()) - self.distance
    }

    pub fn from_point_normal(point: &Point3, normal: &Vector3) -> Self {
        let n = normal.normalized();
        Self {
            normal: n,
            distance: n.dot(&point.to_vec3()),
        }
    }
}

/// A sphere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
}

/// A convex hull stored as a point set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvexHull {
    pub points: Vec<Point3>,
}

impl ConvexHull {
    pub fn new(points: Vec<Point3>) -> Self {
        Self { points }
    }

    pub fn bounding_box(&self) -> AABB {
        if self.points.is_empty() {
            return AABB::default();
        }
        let mut min = self.points[0];
        let mut max = self.points[0];
        for p in &self.points[1..] {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        AABB { min, max }
    }

    pub fn centroid(&self) -> Point3 {
        if self.points.is_empty() {
            return Point3::default();
        }
        let n = self.points.len() as f64;
        let (sx, sy, sz) = self
            .points
            .iter()
            .fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
                (ax + p.x, ay + p.y, az + p.z)
            });
        Point3::new(sx / n, sy / n, sz / n)
    }
}

// ---------------------------------------------------------------------------
// Spatial predicates
// ---------------------------------------------------------------------------

/// A spatial predicate over scene entities and regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialPredicate {
    Proximity {
        entity_a: EntityId,
        entity_b: EntityId,
        threshold: f64,
    },
    Inside {
        entity: EntityId,
        region: RegionId,
    },
    Intersection {
        region_a: RegionId,
        region_b: RegionId,
    },
    Containment {
        inner: RegionId,
        outer: RegionId,
    },
    Alignment {
        entity_a: EntityId,
        entity_b: EntityId,
        tolerance: f64,
    },
    GazeAt {
        observer: EntityId,
        target: EntityId,
        cone_angle: f64,
    },
    Separation {
        entity_a: EntityId,
        entity_b: EntityId,
        min_distance: f64,
    },
}

impl Eq for SpatialPredicate {}

impl std::hash::Hash for SpatialPredicate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                threshold,
            } => {
                entity_a.hash(state);
                entity_b.hash(state);
                OrderedFloat(*threshold).hash(state);
            }
            SpatialPredicate::Inside { entity, region } => {
                entity.hash(state);
                region.hash(state);
            }
            SpatialPredicate::Intersection {
                region_a,
                region_b,
            } => {
                region_a.hash(state);
                region_b.hash(state);
            }
            SpatialPredicate::Containment { inner, outer } => {
                inner.hash(state);
                outer.hash(state);
            }
            SpatialPredicate::Alignment {
                entity_a,
                entity_b,
                tolerance,
            } => {
                entity_a.hash(state);
                entity_b.hash(state);
                OrderedFloat(*tolerance).hash(state);
            }
            SpatialPredicate::GazeAt {
                observer,
                target,
                cone_angle,
            } => {
                observer.hash(state);
                target.hash(state);
                OrderedFloat(*cone_angle).hash(state);
            }
            SpatialPredicate::Separation {
                entity_a,
                entity_b,
                min_distance,
            } => {
                entity_a.hash(state);
                entity_b.hash(state);
                OrderedFloat(*min_distance).hash(state);
            }
        }
    }
}

impl SpatialPredicate {
    pub fn involved_entities(&self) -> Vec<EntityId> {
        match self {
            SpatialPredicate::Proximity { entity_a, entity_b, .. } => vec![*entity_a, *entity_b],
            SpatialPredicate::Inside { entity, .. } => vec![*entity],
            SpatialPredicate::Alignment { entity_a, entity_b, .. } => vec![*entity_a, *entity_b],
            SpatialPredicate::GazeAt { observer, target, .. } => vec![*observer, *target],
            SpatialPredicate::Separation { entity_a, entity_b, .. } => vec![*entity_a, *entity_b],
            _ => vec![],
        }
    }
}

/// Valuation mapping predicate ids to boolean values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PredicateValuation {
    pub values: IndexMap<SpatialPredicateId, bool>,
}

impl PredicateValuation {
    pub fn new() -> Self {
        Self {
            values: IndexMap::new(),
        }
    }

    pub fn set(&mut self, id: SpatialPredicateId, val: bool) {
        self.values.insert(id, val);
    }

    pub fn get(&self, id: &SpatialPredicateId) -> Option<bool> {
        self.values.get(id).copied()
    }

    pub fn is_compatible(&self, other: &PredicateValuation) -> bool {
        for (k, v) in &self.values {
            if let Some(ov) = other.values.get(k) {
                if v != ov {
                    return false;
                }
            }
        }
        true
    }

    pub fn merge(&self, other: &PredicateValuation) -> Option<PredicateValuation> {
        if !self.is_compatible(other) {
            return None;
        }
        let mut result = self.clone();
        for (k, v) in &other.values {
            result.values.insert(*k, *v);
        }
        Some(result)
    }
}

/// A spatial constraint combining predicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpatialConstraint {
    Predicate(SpatialPredicateId),
    Not(Box<SpatialConstraint>),
    And(Box<SpatialConstraint>, Box<SpatialConstraint>),
    Or(Box<SpatialConstraint>, Box<SpatialConstraint>),
    Implies(Box<SpatialConstraint>, Box<SpatialConstraint>),
    True,
    False,
}

impl SpatialConstraint {
    pub fn evaluate(&self, valuation: &PredicateValuation) -> Option<bool> {
        match self {
            SpatialConstraint::True => Some(true),
            SpatialConstraint::False => Some(false),
            SpatialConstraint::Predicate(id) => valuation.get(id),
            SpatialConstraint::Not(c) => c.evaluate(valuation).map(|v| !v),
            SpatialConstraint::And(a, b) => {
                match (a.evaluate(valuation), b.evaluate(valuation)) {
                    (Some(va), Some(vb)) => Some(va && vb),
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    _ => None,
                }
            }
            SpatialConstraint::Or(a, b) => {
                match (a.evaluate(valuation), b.evaluate(valuation)) {
                    (Some(va), Some(vb)) => Some(va || vb),
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    _ => None,
                }
            }
            SpatialConstraint::Implies(a, b) => {
                match (a.evaluate(valuation), b.evaluate(valuation)) {
                    (Some(va), Some(vb)) => Some(!va || vb),
                    (Some(false), _) => Some(true),
                    (_, Some(true)) => Some(true),
                    _ => None,
                }
            }
        }
    }

    pub fn referenced_predicates(&self) -> Vec<SpatialPredicateId> {
        match self {
            SpatialConstraint::True | SpatialConstraint::False => vec![],
            SpatialConstraint::Predicate(id) => vec![*id],
            SpatialConstraint::Not(c) => c.referenced_predicates(),
            SpatialConstraint::And(a, b)
            | SpatialConstraint::Or(a, b)
            | SpatialConstraint::Implies(a, b) => {
                let mut preds = a.referenced_predicates();
                preds.extend(b.referenced_predicates());
                preds.sort();
                preds.dedup();
                preds
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Automaton types
// ---------------------------------------------------------------------------

/// Guard on a transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Guard {
    True,
    Predicate(SpatialPredicateId),
    And(Box<Guard>, Box<Guard>),
    Or(Box<Guard>, Box<Guard>),
    Not(Box<Guard>),
}

impl Guard {
    pub fn evaluate(&self, valuation: &PredicateValuation) -> Option<bool> {
        match self {
            Guard::True => Some(true),
            Guard::Predicate(id) => valuation.get(id),
            Guard::And(a, b) => match (a.evaluate(valuation), b.evaluate(valuation)) {
                (Some(va), Some(vb)) => Some(va && vb),
                (Some(false), _) | (_, Some(false)) => Some(false),
                _ => None,
            },
            Guard::Or(a, b) => match (a.evaluate(valuation), b.evaluate(valuation)) {
                (Some(va), Some(vb)) => Some(va || vb),
                (Some(true), _) | (_, Some(true)) => Some(true),
                _ => None,
            },
            Guard::Not(g) => g.evaluate(valuation).map(|v| !v),
        }
    }

    pub fn referenced_predicates(&self) -> Vec<SpatialPredicateId> {
        match self {
            Guard::True => vec![],
            Guard::Predicate(id) => vec![*id],
            Guard::And(a, b) | Guard::Or(a, b) => {
                let mut preds = a.referenced_predicates();
                preds.extend(b.referenced_predicates());
                preds.sort();
                preds.dedup();
                preds
            }
            Guard::Not(g) => g.referenced_predicates(),
        }
    }

    pub fn to_constraint(&self) -> SpatialConstraint {
        match self {
            Guard::True => SpatialConstraint::True,
            Guard::Predicate(id) => SpatialConstraint::Predicate(*id),
            Guard::And(a, b) => SpatialConstraint::And(
                Box::new(a.to_constraint()),
                Box::new(b.to_constraint()),
            ),
            Guard::Or(a, b) => SpatialConstraint::Or(
                Box::new(a.to_constraint()),
                Box::new(b.to_constraint()),
            ),
            Guard::Not(g) => SpatialConstraint::Not(Box::new(g.to_constraint())),
        }
    }
}

/// Value type for variable assignments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

/// Action on a transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Noop,
    Assign(String, Value),
    Emit(String),
    Sequence(Vec<Action>),
}

/// A transition in the automaton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
    pub action: Action,
}

/// A state in the automaton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub name: String,
    pub invariant: Option<Guard>,
    pub is_accepting: bool,
}

/// Full automaton definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutomatonDef {
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,
    pub initial: StateId,
    pub accepting: Vec<StateId>,
    pub predicates: IndexMap<SpatialPredicateId, SpatialPredicate>,
}

impl AutomatonDef {
    pub fn state_by_id(&self, id: StateId) -> Option<&State> {
        self.states.iter().find(|s| s.id == id)
    }

    pub fn transitions_from(&self, source: StateId) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.source == source)
            .collect()
    }

    pub fn transitions_to(&self, target: StateId) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.target == target)
            .collect()
    }

    pub fn successor_states(&self, state: StateId) -> Vec<StateId> {
        self.transitions_from(state)
            .iter()
            .map(|t| t.target)
            .collect()
    }

    pub fn predecessor_states(&self, state: StateId) -> Vec<StateId> {
        self.transitions_to(state)
            .iter()
            .map(|t| t.source)
            .collect()
    }

    pub fn state_ids(&self) -> Vec<StateId> {
        self.states.iter().map(|s| s.id).collect()
    }

    pub fn is_accepting(&self, state: StateId) -> bool {
        self.accepting.contains(&state)
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }
}

// ---------------------------------------------------------------------------
// Scene configuration
// ---------------------------------------------------------------------------

/// A scene entity with spatial extent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: EntityId,
    pub name: String,
    pub position: Point3,
    pub bounding_box: AABB,
}

/// Full scene configuration at a point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneConfiguration {
    pub entities: Vec<SceneEntity>,
    pub regions: IndexMap<RegionId, AABB>,
    pub predicate_defs: IndexMap<SpatialPredicateId, SpatialPredicate>,
}

impl SceneConfiguration {
    pub fn evaluate_predicate(&self, pred: &SpatialPredicate) -> bool {
        match pred {
            SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                threshold,
            } => {
                let a = self.entities.iter().find(|e| e.id == *entity_a);
                let b = self.entities.iter().find(|e| e.id == *entity_b);
                match (a, b) {
                    (Some(ea), Some(eb)) => ea.position.distance_to(&eb.position) <= *threshold,
                    _ => false,
                }
            }
            SpatialPredicate::Inside { entity, region } => {
                let e = self.entities.iter().find(|e| e.id == *entity);
                let r = self.regions.get(region);
                match (e, r) {
                    (Some(ent), Some(reg)) => reg.contains_point(&ent.position),
                    _ => false,
                }
            }
            SpatialPredicate::Intersection {
                region_a,
                region_b,
            } => {
                let a = self.regions.get(region_a);
                let b = self.regions.get(region_b);
                match (a, b) {
                    (Some(ra), Some(rb)) => ra.intersects(rb),
                    _ => false,
                }
            }
            SpatialPredicate::Containment { inner, outer } => {
                let i = self.regions.get(inner);
                let o = self.regions.get(outer);
                match (i, o) {
                    (Some(ri), Some(ro)) => ro.contains_aabb(ri),
                    _ => false,
                }
            }
            SpatialPredicate::Alignment {
                entity_a,
                entity_b,
                tolerance,
            } => {
                let a = self.entities.iter().find(|e| e.id == *entity_a);
                let b = self.entities.iter().find(|e| e.id == *entity_b);
                match (a, b) {
                    (Some(ea), Some(eb)) => {
                        let dy = (ea.position.y - eb.position.y).abs();
                        let dz = (ea.position.z - eb.position.z).abs();
                        dy <= *tolerance && dz <= *tolerance
                    }
                    _ => false,
                }
            }
            SpatialPredicate::GazeAt {
                observer,
                target,
                cone_angle,
            } => {
                let o = self.entities.iter().find(|e| e.id == *observer);
                let t = self.entities.iter().find(|e| e.id == *target);
                match (o, t) {
                    (Some(obs), Some(tgt)) => {
                        let dir = Vector3::new(
                            tgt.position.x - obs.position.x,
                            tgt.position.y - obs.position.y,
                            tgt.position.z - obs.position.z,
                        );
                        if dir.length() < 1e-12 {
                            return true;
                        }
                        let forward = Vector3::new(1.0, 0.0, 0.0);
                        let cos_angle = dir.normalized().dot(&forward);
                        cos_angle >= cone_angle.cos()
                    }
                    _ => false,
                }
            }
            SpatialPredicate::Separation {
                entity_a,
                entity_b,
                min_distance,
            } => {
                let a = self.entities.iter().find(|e| e.id == *entity_a);
                let b = self.entities.iter().find(|e| e.id == *entity_b);
                match (a, b) {
                    (Some(ea), Some(eb)) => {
                        ea.position.distance_to(&eb.position) >= *min_distance
                    }
                    _ => false,
                }
            }
        }
    }

    pub fn evaluate_all(&self) -> PredicateValuation {
        let mut val = PredicateValuation::new();
        for (id, pred) in &self.predicate_defs {
            val.set(*id, self.evaluate_predicate(pred));
        }
        val
    }
}

// ---------------------------------------------------------------------------
// GJK / EPA local equivalents (simplified)
// ---------------------------------------------------------------------------

/// Result of a GJK distance query between two convex shapes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GjkResult {
    pub distance: f64,
    pub closest_a: Point3,
    pub closest_b: Point3,
    pub intersecting: bool,
}

/// Support function: furthest point of an AABB in a given direction.
pub fn aabb_support(aabb: &AABB, direction: &Vector3) -> Point3 {
    Point3 {
        x: if direction.x >= 0.0 {
            aabb.max.x
        } else {
            aabb.min.x
        },
        y: if direction.y >= 0.0 {
            aabb.max.y
        } else {
            aabb.min.y
        },
        z: if direction.z >= 0.0 {
            aabb.max.z
        } else {
            aabb.min.z
        },
    }
}

/// Simplified GJK distance between two AABBs.
pub fn gjk_distance_aabb(a: &AABB, b: &AABB) -> GjkResult {
    let mut dx = 0.0f64;
    let mut dy = 0.0f64;
    let mut dz = 0.0f64;

    if a.max.x < b.min.x {
        dx = b.min.x - a.max.x;
    } else if b.max.x < a.min.x {
        dx = a.min.x - b.max.x;
    }
    if a.max.y < b.min.y {
        dy = b.min.y - a.max.y;
    } else if b.max.y < a.min.y {
        dy = a.min.y - b.max.y;
    }
    if a.max.z < b.min.z {
        dz = b.min.z - a.max.z;
    } else if b.max.z < a.min.z {
        dz = a.min.z - b.max.z;
    }

    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let intersecting = dist < 1e-12;

    GjkResult {
        distance: dist,
        closest_a: a.center(),
        closest_b: b.center(),
        intersecting,
    }
}

/// EPA penetration depth for two overlapping AABBs.
pub fn epa_penetration_aabb(a: &AABB, b: &AABB) -> f64 {
    let overlap_x = (a.max.x.min(b.max.x) - a.min.x.max(b.min.x)).max(0.0);
    let overlap_y = (a.max.y.min(b.max.y) - a.min.y.max(b.min.y)).max(0.0);
    let overlap_z = (a.max.z.min(b.max.z) - a.min.z.max(b.min.z)).max(0.0);
    overlap_x.min(overlap_y).min(overlap_z)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the CEGAR engine.
#[derive(Debug, thiserror::Error)]
pub enum CegarError {
    #[error("Timeout after {iterations} iterations")]
    Timeout { iterations: usize },

    #[error("Resource exhausted: {reason}")]
    ResourceExhausted { reason: String },

    #[error("Invalid property specification: {0}")]
    InvalidProperty(String),

    #[error("Abstraction refinement failed: {0}")]
    RefinementFailed(String),

    #[error("Model checking error: {0}")]
    ModelCheckError(String),

    #[error("Counterexample analysis error: {0}")]
    CounterexampleError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid certificate: {0}")]
    InvalidCertificate(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for CegarError {
    fn from(e: serde_json::Error) -> Self {
        CegarError::SerializationError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use abstraction::{
    AbstractBlock, AbstractDomain, AbstractTransitionRelation, AbstractionState,
    GeometricAbstraction, PartitionRefinement, SpatialPartition,
};
pub use cegar_loop::{CEGARConfig, CEGARStatistics, CEGARVerifier, RefinementStrategy};
pub use certificate::{
    CertificateBuilder, CounterexampleCertificate, ProofCertificate, VerificationCertificate,
};
pub use compositional::CompositionalVerifier;
pub use counterexample::{
    ConcreteCounterexample, Counterexample, FeasibilityResult, InfeasibilityWitness,
    RefinementHint,
};
pub use model_checker::ModelChecker;
pub use properties::Property;
pub use pruning::GeometricPruner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_contains() {
        let aabb = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        assert!(aabb.contains_point(&Point3::new(5.0, 5.0, 5.0)));
        assert!(!aabb.contains_point(&Point3::new(15.0, 5.0, 5.0)));
    }

    #[test]
    fn test_aabb_split() {
        let aabb = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let (left, right) = aabb.split_at_axis(0, 5.0);
        assert!((left.max.x - 5.0).abs() < 1e-10);
        assert!((right.min.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_intersection() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 5.0, 5.0));
        let b = AABB::new(Point3::new(3.0, 3.0, 3.0), Point3::new(8.0, 8.0, 8.0));
        let i = a.intersection(&b).unwrap();
        assert!((i.min.x - 3.0).abs() < 1e-10);
        assert!((i.max.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_predicate_valuation() {
        let mut v1 = PredicateValuation::new();
        v1.set(SpatialPredicateId(0), true);
        v1.set(SpatialPredicateId(1), false);
        let mut v2 = PredicateValuation::new();
        v2.set(SpatialPredicateId(0), true);
        v2.set(SpatialPredicateId(2), true);
        assert!(v1.is_compatible(&v2));
        let merged = v1.merge(&v2).unwrap();
        assert_eq!(merged.values.len(), 3);
    }

    #[test]
    fn test_predicate_valuation_incompatible() {
        let mut v1 = PredicateValuation::new();
        v1.set(SpatialPredicateId(0), true);
        let mut v2 = PredicateValuation::new();
        v2.set(SpatialPredicateId(0), false);
        assert!(!v1.is_compatible(&v2));
        assert!(v1.merge(&v2).is_none());
    }

    #[test]
    fn test_guard_evaluate() {
        let mut val = PredicateValuation::new();
        val.set(SpatialPredicateId(0), true);
        val.set(SpatialPredicateId(1), false);
        let g = Guard::And(
            Box::new(Guard::Predicate(SpatialPredicateId(0))),
            Box::new(Guard::Not(Box::new(Guard::Predicate(SpatialPredicateId(
                1,
            ))))),
        );
        assert_eq!(g.evaluate(&val), Some(true));
    }

    #[test]
    fn test_gjk_distance() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AABB::new(Point3::new(3.0, 0.0, 0.0), Point3::new(4.0, 1.0, 1.0));
        let r = gjk_distance_aabb(&a, &b);
        assert!((r.distance - 2.0).abs() < 1e-10);
        assert!(!r.intersecting);
    }

    #[test]
    fn test_gjk_intersecting() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));
        let b = AABB::new(Point3::new(1.0, 1.0, 1.0), Point3::new(3.0, 3.0, 3.0));
        let r = gjk_distance_aabb(&a, &b);
        assert!(r.intersecting);
    }

    #[test]
    fn test_spatial_constraint_eval() {
        let mut val = PredicateValuation::new();
        val.set(SpatialPredicateId(0), true);
        val.set(SpatialPredicateId(1), true);
        let c = SpatialConstraint::Implies(
            Box::new(SpatialConstraint::Predicate(SpatialPredicateId(0))),
            Box::new(SpatialConstraint::Predicate(SpatialPredicateId(1))),
        );
        assert_eq!(c.evaluate(&val), Some(true));
    }

    #[test]
    fn test_automaton_successors() {
        let aut = AutomatonDef {
            states: vec![
                State {
                    id: StateId(0),
                    name: "s0".into(),
                    invariant: None,
                    is_accepting: false,
                },
                State {
                    id: StateId(1),
                    name: "s1".into(),
                    invariant: None,
                    is_accepting: true,
                },
            ],
            transitions: vec![Transition {
                id: TransitionId(0),
                source: StateId(0),
                target: StateId(1),
                guard: Guard::True,
                action: Action::Noop,
            }],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: IndexMap::new(),
        };
        assert_eq!(aut.successor_states(StateId(0)), vec![StateId(1)]);
        assert!(aut.is_accepting(StateId(1)));
    }

    #[test]
    fn test_scene_evaluate_proximity() {
        let scene = SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "a".into(),
                    position: Point3::new(0.0, 0.0, 0.0),
                    bounding_box: AABB::default(),
                },
                SceneEntity {
                    id: EntityId(1),
                    name: "b".into(),
                    position: Point3::new(1.0, 0.0, 0.0),
                    bounding_box: AABB::default(),
                },
            ],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
        };
        let pred = SpatialPredicate::Proximity {
            entity_a: EntityId(0),
            entity_b: EntityId(1),
            threshold: 2.0,
        };
        assert!(scene.evaluate_predicate(&pred));
    }

    #[test]
    fn test_vector3_operations() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.z - 1.0).abs() < 1e-10);
        assert!((a.dot(&b)).abs() < 1e-10);
    }

    #[test]
    fn test_epa_penetration() {
        let a = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 3.0, 3.0));
        let b = AABB::new(Point3::new(2.0, 2.0, 2.0), Point3::new(5.0, 5.0, 5.0));
        let depth = epa_penetration_aabb(&a, &b);
        assert!((depth - 1.0).abs() < 1e-10);
    }
}
