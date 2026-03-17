//! Spatial predicate types for scene reasoning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::geometry::{AABB, Transform3D};

/// Unique identifier for an entity in the scene.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Unique identifier for a spatial region.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub String);

/// Unique identifier for an interaction zone.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZoneId(pub String);

/// Unique identifier for a spatial predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialPredicateId(pub String);

/// A spatial predicate describing a geometric relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    GazeAt {
        source: EntityId,
        target: EntityId,
        cone_angle: f64,
    },
    Containment {
        inner: RegionId,
        outer: RegionId,
    },
    Intersection {
        region_a: RegionId,
        region_b: RegionId,
    },
    Touching {
        entity_a: EntityId,
        entity_b: EntityId,
    },
}

impl SpatialPredicate {
    pub fn involved_entities(&self) -> Vec<&EntityId> {
        match self {
            SpatialPredicate::Proximity { entity_a, entity_b, .. } => vec![entity_a, entity_b],
            SpatialPredicate::Inside { entity, .. } => vec![entity],
            SpatialPredicate::GazeAt { source, target, .. } => vec![source, target],
            SpatialPredicate::Touching { entity_a, entity_b } => vec![entity_a, entity_b],
            _ => vec![],
        }
    }

    pub fn involved_regions(&self) -> Vec<&RegionId> {
        match self {
            SpatialPredicate::Inside { region, .. } => vec![region],
            SpatialPredicate::Containment { inner, outer } => vec![inner, outer],
            SpatialPredicate::Intersection { region_a, region_b } => vec![region_a, region_b],
            _ => vec![],
        }
    }
}

/// Valuation of spatial predicates (maps predicate IDs to boolean values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateValuation {
    pub values: HashMap<SpatialPredicateId, bool>,
}

impl PredicateValuation {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn set(&mut self, id: SpatialPredicateId, value: bool) {
        self.values.insert(id, value);
    }

    pub fn get(&self, id: &SpatialPredicateId) -> Option<bool> {
        self.values.get(id).copied()
    }
}

impl Default for PredicateValuation {
    fn default() -> Self {
        Self::new()
    }
}

/// A spatial constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConstraint {
    pub predicate_id: SpatialPredicateId,
    pub required_value: bool,
}

/// A spatial region definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialRegion {
    Aabb(AABB),
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    ConvexHull {
        points: Vec<[f64; 3]>,
    },
    Composite {
        regions: Vec<SpatialRegion>,
    },
}

impl SpatialRegion {
    pub fn to_aabb(&self) -> AABB {
        match self {
            SpatialRegion::Aabb(aabb) => *aabb,
            SpatialRegion::Sphere { center, radius } => AABB::new(
                [center[0] - radius, center[1] - radius, center[2] - radius],
                [center[0] + radius, center[1] + radius, center[2] + radius],
            ),
            SpatialRegion::ConvexHull { points } => {
                let pts: Vec<_> = points
                    .iter()
                    .map(|p| nalgebra::Point3::new(p[0], p[1], p[2]))
                    .collect();
                AABB::from_points(&pts)
            }
            SpatialRegion::Composite { regions } => {
                let mut result = AABB::empty();
                for r in regions {
                    result = result.merge(&r.to_aabb());
                }
                result
            }
        }
    }
}

/// An entity in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: EntityId,
    pub transform: Transform3D,
    pub bounds: AABB,
    pub region: Option<SpatialRegion>,
}

/// Complete scene configuration with all entities and regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneConfiguration {
    pub entities: HashMap<EntityId, SceneEntity>,
    pub regions: HashMap<RegionId, SpatialRegion>,
    pub zones: HashMap<ZoneId, Vec<RegionId>>,
}

impl SceneConfiguration {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            regions: HashMap::new(),
            zones: HashMap::new(),
        }
    }
}

impl Default for SceneConfiguration {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for evaluating spatial predicates against a scene.
pub trait SpatialPredicateEvaluator {
    fn evaluate(
        &self,
        predicate: &SpatialPredicate,
        scene: &SceneConfiguration,
    ) -> bool;

    fn evaluate_all(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        scene: &SceneConfiguration,
    ) -> PredicateValuation;
}
