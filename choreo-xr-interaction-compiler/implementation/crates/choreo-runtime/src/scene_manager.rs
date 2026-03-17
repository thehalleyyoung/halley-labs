//! Scene state management: entity registry, spatial indexing, predicate
//! evaluation, dirty tracking, and predicate caching.

use choreo_automata::{
    EntityId, Point3, RegionId, SceneConfiguration, SceneEntity, Sphere, SpatialPredicate,
    SpatialRegion, TimePoint, AABB,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// EntityState
// ---------------------------------------------------------------------------

/// Tracked state for a single entity.
#[derive(Debug, Clone)]
struct EntityState {
    entity: SceneEntity,
    version: u64,
}

// ---------------------------------------------------------------------------
// SceneSnapshot
// ---------------------------------------------------------------------------

/// Immutable snapshot of the scene at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSnapshot {
    pub timestamp: f64,
    pub entity_count: usize,
    pub region_count: usize,
    pub configuration: SceneConfiguration,
}

// ---------------------------------------------------------------------------
// PredicateCache
// ---------------------------------------------------------------------------

/// Cache for evaluated predicate results.
#[derive(Debug, Clone)]
struct PredicateCache {
    values: HashMap<String, CachedPredicate>,
}

#[derive(Debug, Clone)]
struct CachedPredicate {
    result: bool,
    /// Combined version of all entities/regions involved.
    version: u64,
}

impl PredicateCache {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    fn get(&self, key: &str, current_version: u64) -> Option<bool> {
        self.values.get(key).and_then(|cp| {
            if cp.version == current_version {
                Some(cp.result)
            } else {
                None
            }
        })
    }

    fn insert(&mut self, key: String, result: bool, version: u64) {
        self.values.insert(
            key,
            CachedPredicate { result, version },
        );
    }

    fn invalidate_all(&mut self) {
        self.values.clear();
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// SceneManager
// ---------------------------------------------------------------------------

/// Manages the spatial scene: entity registry, region definitions, dirty
/// tracking for incremental evaluation, and predicate caching.
#[derive(Debug)]
pub struct SceneManager {
    /// Entity registry.
    entities: HashMap<String, EntityState>,
    /// Region definitions.
    regions: HashMap<String, SpatialRegion>,
    /// Global version counter.
    version: u64,
    /// Set of entity ids that changed since the last snapshot.
    dirty_entities: HashSet<String>,
    /// Set of region ids that changed since the last snapshot.
    dirty_regions: HashSet<String>,
    /// Predicate cache.
    cache: PredicateCache,
    /// Current simulation time.
    current_time: f64,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            regions: HashMap::new(),
            version: 0,
            dirty_entities: HashSet::new(),
            dirty_regions: HashSet::new(),
            cache: PredicateCache::new(),
            current_time: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Entity management
    // -----------------------------------------------------------------------

    /// Add an entity to the scene.
    pub fn add_entity(&mut self, id: impl Into<String>, position: [f64; 3]) {
        let id = id.into();
        self.version += 1;
        let entity = SceneEntity {
            id: EntityId(id.clone()),
            position: Point3::new(position[0], position[1], position[2]),
            bounding: Some(AABB::new(
                Point3::new(
                    position[0] - 0.5,
                    position[1] - 0.5,
                    position[2] - 0.5,
                ),
                Point3::new(
                    position[0] + 0.5,
                    position[1] + 0.5,
                    position[2] + 0.5,
                ),
            )),
            properties: HashMap::new(),
        };
        self.entities.insert(
            id.clone(),
            EntityState {
                entity,
                version: self.version,
            },
        );
        self.dirty_entities.insert(id);
    }

    /// Remove an entity from the scene.
    pub fn remove_entity(&mut self, id: &str) -> bool {
        if self.entities.remove(id).is_some() {
            self.version += 1;
            self.dirty_entities.insert(id.to_string());
            true
        } else {
            false
        }
    }

    /// Update an entity's position (and optionally rotation — simplified to
    /// just position here).
    pub fn update_entity(
        &mut self,
        id: &str,
        position: [f64; 3],
        _rotation: Option<[f64; 4]>,
    ) -> bool {
        if let Some(state) = self.entities.get_mut(id) {
            self.version += 1;
            state.entity.position =
                Point3::new(position[0], position[1], position[2]);
            state.entity.bounding = Some(AABB::new(
                Point3::new(
                    position[0] - 0.5,
                    position[1] - 0.5,
                    position[2] - 0.5,
                ),
                Point3::new(
                    position[0] + 0.5,
                    position[1] + 0.5,
                    position[2] + 0.5,
                ),
            ));
            state.version = self.version;
            self.dirty_entities.insert(id.to_string());
            true
        } else {
            false
        }
    }

    /// Set a property on an entity.
    pub fn set_entity_property(
        &mut self,
        id: &str,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> bool {
        if let Some(state) = self.entities.get_mut(id) {
            state.entity.properties.insert(key.into(), value.into());
            self.version += 1;
            state.version = self.version;
            self.dirty_entities.insert(id.to_string());
            true
        } else {
            false
        }
    }

    /// Get an entity's current position.
    pub fn entity_position(&self, id: &str) -> Option<[f64; 3]> {
        self.entities.get(id).map(|s| {
            let p = &s.entity.position;
            [p.x, p.y, p.z]
        })
    }

    /// Check if an entity exists.
    pub fn has_entity(&self, id: &str) -> bool {
        self.entities.contains_key(id)
    }

    /// Number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // -----------------------------------------------------------------------
    // Region management
    // -----------------------------------------------------------------------

    /// Add or update a region definition.
    pub fn set_region(&mut self, id: impl Into<String>, region: SpatialRegion) {
        let id = id.into();
        self.version += 1;
        self.regions.insert(id.clone(), region);
        self.dirty_regions.insert(id);
    }

    /// Remove a region.
    pub fn remove_region(&mut self, id: &str) -> bool {
        if self.regions.remove(id).is_some() {
            self.version += 1;
            self.dirty_regions.insert(id.to_string());
            true
        } else {
            false
        }
    }

    /// Number of regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    // -----------------------------------------------------------------------
    // Spatial predicate queries
    // -----------------------------------------------------------------------

    /// Evaluate a spatial predicate against the current scene.
    pub fn query_spatial_predicate(&mut self, predicate: &SpatialPredicate) -> bool {
        let key = format!("{:?}", predicate);
        let dep_version = self.dependency_version(predicate);

        // Check cache
        if let Some(cached) = self.cache.get(&key, dep_version) {
            return cached;
        }

        // Evaluate
        let config = self.to_scene_configuration();
        let result = evaluate_predicate(predicate, &config);

        self.cache.insert(key, result, dep_version);
        result
    }

    /// Batch-evaluate multiple predicates.
    pub fn query_predicates(
        &mut self,
        predicates: &[(String, SpatialPredicate)],
    ) -> HashMap<String, bool> {
        let mut results = HashMap::new();
        for (name, pred) in predicates {
            let result = self.query_spatial_predicate(pred);
            results.insert(name.clone(), result);
        }
        results
    }

    /// Get the combined version of entities/regions involved in a predicate.
    fn dependency_version(&self, predicate: &SpatialPredicate) -> u64 {
        let mut max_ver = 0u64;
        match predicate {
            SpatialPredicate::Inside { entity, region } => {
                if let Some(s) = self.entities.get(&entity.0) {
                    max_ver = max_ver.max(s.version);
                }
                // Region version tracked globally
                max_ver = max_ver.max(self.version);
            }
            SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                ..
            } => {
                if let Some(s) = self.entities.get(&entity_a.0) {
                    max_ver = max_ver.max(s.version);
                }
                if let Some(s) = self.entities.get(&entity_b.0) {
                    max_ver = max_ver.max(s.version);
                }
            }
            SpatialPredicate::Contact {
                entity_a,
                entity_b,
            } => {
                if let Some(s) = self.entities.get(&entity_a.0) {
                    max_ver = max_ver.max(s.version);
                }
                if let Some(s) = self.entities.get(&entity_b.0) {
                    max_ver = max_ver.max(s.version);
                }
            }
            SpatialPredicate::GazeAt { entity, .. } => {
                if let Some(s) = self.entities.get(&entity.0) {
                    max_ver = max_ver.max(s.version);
                }
            }
            SpatialPredicate::Grasping { hand, object } => {
                if let Some(s) = self.entities.get(&hand.0) {
                    max_ver = max_ver.max(s.version);
                }
                if let Some(s) = self.entities.get(&object.0) {
                    max_ver = max_ver.max(s.version);
                }
            }
            SpatialPredicate::And(preds) | SpatialPredicate::Or(preds) => {
                for p in preds {
                    max_ver = max_ver.max(self.dependency_version(p));
                }
            }
            SpatialPredicate::Not(inner) => {
                max_ver = max_ver.max(self.dependency_version(inner));
            }
            SpatialPredicate::Named(_) => {
                max_ver = self.version;
            }
        }
        max_ver
    }

    // -----------------------------------------------------------------------
    // Snapshots and dirty tracking
    // -----------------------------------------------------------------------

    /// Get a snapshot of the current scene.
    pub fn get_snapshot(&self) -> SceneSnapshot {
        SceneSnapshot {
            timestamp: self.current_time,
            entity_count: self.entities.len(),
            region_count: self.regions.len(),
            configuration: self.to_scene_configuration(),
        }
    }

    /// Build a `SceneConfiguration` from the current state.
    pub fn to_scene_configuration(&self) -> SceneConfiguration {
        SceneConfiguration {
            entities: self
                .entities
                .values()
                .map(|s| s.entity.clone())
                .collect(),
            regions: self
                .regions
                .iter()
                .map(|(k, v)| (RegionId(k.clone()), v.clone()))
                .collect(),
            timestamp: TimePoint(self.current_time),
        }
    }

    /// Get the set of dirty entity ids and clear the dirty set.
    pub fn take_dirty_entities(&mut self) -> HashSet<String> {
        std::mem::take(&mut self.dirty_entities)
    }

    /// Get the set of dirty region ids and clear the dirty set.
    pub fn take_dirty_regions(&mut self) -> HashSet<String> {
        std::mem::take(&mut self.dirty_regions)
    }

    /// Check if any entity or region has changed.
    pub fn is_dirty(&self) -> bool {
        !self.dirty_entities.is_empty() || !self.dirty_regions.is_empty()
    }

    /// Invalidate all caches.
    pub fn invalidate_cache(&mut self) {
        self.cache.invalidate_all();
    }

    /// Number of cached predicate results.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Set the current simulation time.
    pub fn set_time(&mut self, time: f64) {
        self.current_time = time;
    }

    /// Get the current simulation time.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Predicate evaluation
// ---------------------------------------------------------------------------

fn evaluate_predicate(
    predicate: &SpatialPredicate,
    scene: &SceneConfiguration,
) -> bool {
    match predicate {
        SpatialPredicate::Inside { entity, region } => {
            let ent = scene.entity(entity);
            let reg = scene.regions.get(region);
            match (ent, reg) {
                (Some(e), Some(r)) => point_in_region(&e.position, r),
                _ => false,
            }
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => {
            let a = scene.entity(entity_a);
            let b = scene.entity(entity_b);
            match (a, b) {
                (Some(ea), Some(eb)) => ea.position.distance_to(&eb.position) <= *threshold,
                _ => false,
            }
        }
        SpatialPredicate::Contact {
            entity_a,
            entity_b,
        } => {
            let a = scene.entity(entity_a);
            let b = scene.entity(entity_b);
            match (a, b) {
                (Some(ea), Some(eb)) => {
                    match (&ea.bounding, &eb.bounding) {
                        (Some(ba), Some(bb)) => ba.intersects(bb),
                        _ => ea.position.distance_to(&eb.position) <= 1.0,
                    }
                }
                _ => false,
            }
        }
        SpatialPredicate::GazeAt { entity, target } => {
            // Simplified: just check if entity faces the target region
            let ent = scene.entity(entity);
            let reg = scene.regions.get(target);
            match (ent, reg) {
                (Some(_), Some(_)) => true, // Simplified
                _ => false,
            }
        }
        SpatialPredicate::Grasping { hand, object } => {
            // Simplified: proximity check
            let h = scene.entity(hand);
            let o = scene.entity(object);
            match (h, o) {
                (Some(eh), Some(eo)) => eh.position.distance_to(&eo.position) <= 0.3,
                _ => false,
            }
        }
        SpatialPredicate::Not(inner) => !evaluate_predicate(inner, scene),
        SpatialPredicate::And(preds) => preds.iter().all(|p| evaluate_predicate(p, scene)),
        SpatialPredicate::Or(preds) => preds.iter().any(|p| evaluate_predicate(p, scene)),
        SpatialPredicate::Named(_) => false, // Unknown named predicate
    }
}

fn point_in_region(point: &Point3, region: &SpatialRegion) -> bool {
    match region {
        SpatialRegion::AABB(aabb) => aabb.contains_point(point),
        SpatialRegion::Sphere(sphere) => {
            point.distance_to(&sphere.center) <= sphere.radius
        }
        SpatialRegion::Named(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_query_entity() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [1.0, 2.0, 3.0]);
        assert!(mgr.has_entity("e1"));
        assert_eq!(mgr.entity_count(), 1);
        assert_eq!(mgr.entity_position("e1"), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn remove_entity() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        assert!(mgr.remove_entity("e1"));
        assert!(!mgr.has_entity("e1"));
        assert!(!mgr.remove_entity("e1"));
    }

    #[test]
    fn update_entity_position() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        assert!(mgr.update_entity("e1", [5.0, 5.0, 5.0], None));
        assert_eq!(mgr.entity_position("e1"), Some([5.0, 5.0, 5.0]));
    }

    #[test]
    fn update_nonexistent_fails() {
        let mut mgr = SceneManager::new();
        assert!(!mgr.update_entity("nope", [0.0, 0.0, 0.0], None));
    }

    #[test]
    fn proximity_predicate() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.add_entity("e2", [1.0, 0.0, 0.0]);

        let pred = SpatialPredicate::Proximity {
            entity_a: EntityId("e1".into()),
            entity_b: EntityId("e2".into()),
            threshold: 2.0,
        };
        assert!(mgr.query_spatial_predicate(&pred));

        let pred_far = SpatialPredicate::Proximity {
            entity_a: EntityId("e1".into()),
            entity_b: EntityId("e2".into()),
            threshold: 0.5,
        };
        assert!(!mgr.query_spatial_predicate(&pred_far));
    }

    #[test]
    fn inside_region_predicate() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.set_region(
            "r1",
            SpatialRegion::AABB(AABB::new(
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, 1.0, 1.0),
            )),
        );

        let pred = SpatialPredicate::Inside {
            entity: EntityId("e1".into()),
            region: RegionId("r1".into()),
        };
        assert!(mgr.query_spatial_predicate(&pred));

        // Move entity outside
        mgr.update_entity("e1", [5.0, 5.0, 5.0], None);
        mgr.invalidate_cache();
        assert!(!mgr.query_spatial_predicate(&pred));
    }

    #[test]
    fn predicate_cache_hit() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.add_entity("e2", [1.0, 0.0, 0.0]);

        let pred = SpatialPredicate::Proximity {
            entity_a: EntityId("e1".into()),
            entity_b: EntityId("e2".into()),
            threshold: 2.0,
        };

        // First eval: miss
        let r1 = mgr.query_spatial_predicate(&pred);
        assert!(r1);
        assert_eq!(mgr.cache_size(), 1);

        // Second eval: should use cache
        let r2 = mgr.query_spatial_predicate(&pred);
        assert_eq!(r1, r2);
    }

    #[test]
    fn dirty_tracking() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        assert!(mgr.is_dirty());

        let dirty = mgr.take_dirty_entities();
        assert!(dirty.contains("e1"));
        assert!(!mgr.is_dirty());

        mgr.update_entity("e1", [1.0, 1.0, 1.0], None);
        assert!(mgr.is_dirty());
    }

    #[test]
    fn snapshot() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.set_region(
            "r1",
            SpatialRegion::Sphere(Sphere {
                center: Point3::new(0.0, 0.0, 0.0),
                radius: 5.0,
            }),
        );
        mgr.set_time(42.0);

        let snap = mgr.get_snapshot();
        assert_eq!(snap.entity_count, 1);
        assert_eq!(snap.region_count, 1);
        assert_eq!(snap.timestamp, 42.0);
    }

    #[test]
    fn to_scene_configuration() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [1.0, 2.0, 3.0]);
        let config = mgr.to_scene_configuration();
        assert_eq!(config.entities.len(), 1);
        assert_eq!(config.entities[0].id, EntityId("e1".into()));
    }

    #[test]
    fn region_management() {
        let mut mgr = SceneManager::new();
        mgr.set_region(
            "r1",
            SpatialRegion::AABB(AABB::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 1.0),
            )),
        );
        assert_eq!(mgr.region_count(), 1);
        assert!(mgr.remove_region("r1"));
        assert_eq!(mgr.region_count(), 0);
    }

    #[test]
    fn set_entity_property() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        assert!(mgr.set_entity_property("e1", "color", "red"));
        assert!(!mgr.set_entity_property("nope", "color", "blue"));
    }

    #[test]
    fn contact_predicate() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.add_entity("e2", [0.5, 0.0, 0.0]);

        let pred = SpatialPredicate::Contact {
            entity_a: EntityId("e1".into()),
            entity_b: EntityId("e2".into()),
        };
        // Bounding boxes overlap (each is ±0.5 from center)
        assert!(mgr.query_spatial_predicate(&pred));
    }

    #[test]
    fn negated_predicate() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.add_entity("e2", [100.0, 0.0, 0.0]);

        let pred = SpatialPredicate::Not(Box::new(SpatialPredicate::Proximity {
            entity_a: EntityId("e1".into()),
            entity_b: EntityId("e2".into()),
            threshold: 1.0,
        }));
        assert!(mgr.query_spatial_predicate(&pred));
    }

    #[test]
    fn batch_predicate_evaluation() {
        let mut mgr = SceneManager::new();
        mgr.add_entity("e1", [0.0, 0.0, 0.0]);
        mgr.add_entity("e2", [1.0, 0.0, 0.0]);

        let predicates = vec![
            (
                "close".to_string(),
                SpatialPredicate::Proximity {
                    entity_a: EntityId("e1".into()),
                    entity_b: EntityId("e2".into()),
                    threshold: 2.0,
                },
            ),
            (
                "far".to_string(),
                SpatialPredicate::Proximity {
                    entity_a: EntityId("e1".into()),
                    entity_b: EntityId("e2".into()),
                    threshold: 0.1,
                },
            ),
        ];

        let results = mgr.query_predicates(&predicates);
        assert_eq!(results["close"], true);
        assert_eq!(results["far"], false);
    }
}
