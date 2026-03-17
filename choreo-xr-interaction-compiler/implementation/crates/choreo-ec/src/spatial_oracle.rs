//! Spatial oracle integration.
//!
//! The spatial oracle bridges the geometric world (entity transforms, bounding
//! volumes, regions) and the logical world (spatial predicates that act as
//! fluents in the Event Calculus). It evaluates spatial predicates over scene
//! configurations and detects transitions (predicate value changes) between
//! successive scenes.

use std::collections::{BTreeMap, HashMap};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::fluent::*;
use crate::local_types::*;

// ─── AABB-based spatial index ────────────────────────────────────────────────

/// A simple spatial index node wrapping an entity and its bounding box.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SpatialIndexEntry {
    entity_id: EntityId,
    bounds: AABB,
}

/// A flat spatial index for broad-phase queries.
#[derive(Debug, Clone)]
struct SpatialIndex {
    entries: Vec<SpatialIndexEntry>,
}

#[allow(dead_code)]
impl SpatialIndex {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn insert(&mut self, entity_id: EntityId, bounds: AABB) {
        self.entries.push(SpatialIndexEntry { entity_id, bounds });
    }

    fn build_from_scene(&mut self, scene: &SceneConfiguration) {
        self.clear();
        for (id, entity) in &scene.entities {
            self.insert(*id, entity.bounds);
        }
    }

    fn query_aabb(&self, query: &AABB) -> Vec<EntityId> {
        self.entries
            .iter()
            .filter(|e| e.bounds.intersects(query))
            .map(|e| e.entity_id)
            .collect()
    }

    fn query_sphere(&self, center: &Point3, radius: f64) -> Vec<EntityId> {
        let query = AABB::new(
            Point3::new(center.x - radius, center.y - radius, center.z - radius),
            Point3::new(center.x + radius, center.y + radius, center.z + radius),
        );
        let candidates = self.query_aabb(&query);
        candidates
            .into_iter()
            .filter(|id| {
                self.entries
                    .iter()
                    .find(|e| e.entity_id == *id)
                    .map_or(false, |e| {
                        let entity_center = e.bounds.center();
                        entity_center.distance_to(center) <= radius
                    })
            })
            .collect()
    }

    fn all_pairs(&self) -> Vec<(EntityId, EntityId)> {
        let mut pairs = Vec::new();
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                if self.entries[i].bounds.intersects(&self.entries[j].bounds) {
                    pairs.push((self.entries[i].entity_id, self.entries[j].entity_id));
                }
            }
        }
        pairs
    }
}

// ─── SpatialTransitionEvent ──────────────────────────────────────────────────

/// A spatial predicate changed its truth value between two scenes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialTransitionEvent {
    pub predicate_id: SpatialPredicateId,
    pub predicate: SpatialPredicate,
    pub old_value: bool,
    pub new_value: bool,
    pub time: TimePoint,
}

impl SpatialTransitionEvent {
    /// Whether this transition is a "became true" event.
    pub fn is_onset(&self) -> bool {
        !self.old_value && self.new_value
    }

    /// Whether this transition is a "became false" event.
    pub fn is_offset(&self) -> bool {
        self.old_value && !self.new_value
    }

    /// Convert to an EC event.
    pub fn to_event(&self, event_id: EventId) -> Event {
        Event::new(
            event_id,
            self.time,
            EventKind::SpatialChange {
                predicate_id: self.predicate_id,
                new_value: self.new_value,
            },
        )
    }
}

// ─── SpatialTrajectory ──────────────────────────────────────────────────────

/// A sequence of (time, scene_configuration) pairs representing entity motion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialTrajectory {
    keyframes: BTreeMap<OrderedFloat<f64>, SceneConfiguration>,
}

impl SpatialTrajectory {
    pub fn new() -> Self {
        Self {
            keyframes: BTreeMap::new(),
        }
    }

    /// Add a keyframe at the given time.
    pub fn add_keyframe(&mut self, time: TimePoint, scene: SceneConfiguration) {
        self.keyframes.insert(OrderedFloat(time.0), scene);
    }

    /// Get the scene at an exact keyframe time.
    pub fn keyframe_at(&self, time: TimePoint) -> Option<&SceneConfiguration> {
        self.keyframes.get(&OrderedFloat(time.0))
    }

    /// Interpolate the scene at an arbitrary time between keyframes.
    pub fn interpolate_at(&self, time: TimePoint) -> Option<SceneConfiguration> {
        let key = OrderedFloat(time.0);

        let before = self.keyframes.range(..=key).next_back();
        let after = self.keyframes.range(key..).next();

        match (before, after) {
            (Some((t1, s1)), Some((t2, s2))) if t1 != t2 => {
                let alpha = (time.0 - t1.0) / (t2.0 - t1.0);
                Some(interpolate_scenes(s1, s2, alpha))
            }
            (Some((_, s)), _) => Some(s.clone()),
            (_, Some((_, s))) => Some(s.clone()),
            _ => None,
        }
    }

    /// Number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Time span of the trajectory.
    pub fn time_span(&self) -> Option<TimeInterval> {
        let first = self.keyframes.keys().next()?;
        let last = self.keyframes.keys().next_back()?;
        Some(TimeInterval::new(TimePoint(first.0), TimePoint(last.0)))
    }

    /// Get all keyframe times.
    pub fn keyframe_times(&self) -> Vec<TimePoint> {
        self.keyframes.keys().map(|k| TimePoint(k.0)).collect()
    }

    /// Sample the trajectory at uniform intervals.
    pub fn sample_uniform(&self, delta_t: Duration) -> Vec<(TimePoint, SceneConfiguration)> {
        let span = match self.time_span() {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut samples = Vec::new();
        let mut t = span.start.0;
        while t <= span.end.0 {
            if let Some(scene) = self.interpolate_at(TimePoint(t)) {
                samples.push((TimePoint(t), scene));
            }
            t += delta_t.0;
        }
        samples
    }
}

impl Default for SpatialTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear interpolation between two scene configurations.
fn interpolate_scenes(
    s1: &SceneConfiguration,
    s2: &SceneConfiguration,
    alpha: f64,
) -> SceneConfiguration {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut result = SceneConfiguration::new();

    for (id, e1) in &s1.entities {
        if let Some(e2) = s2.entities.get(id) {
            let transform = e1.transform.lerp(&e2.transform, alpha);
            let mut entity = SceneEntity::new(*id, e1.name.clone(), transform);
            entity.forward = e1.forward.lerp(&e2.forward, alpha);
            entity.bounds = AABB::new(
                e1.bounds.min.lerp(&e2.bounds.min, alpha),
                e1.bounds.max.lerp(&e2.bounds.max, alpha),
            );
            entity.properties = e1.properties.clone();
            result.add_entity(entity);
        } else {
            // Entity only in s1 – fade out (include if alpha < 0.5)
            if alpha < 0.5 {
                result.add_entity(e1.clone());
            }
        }
    }

    // Entities only in s2 – fade in
    for (id, e2) in &s2.entities {
        if !s1.entities.contains_key(id) && alpha >= 0.5 {
            result.add_entity(e2.clone());
        }
    }

    // Regions: take from s1 or s2 depending on alpha
    if alpha < 0.5 {
        result.regions = s1.regions.clone();
    } else {
        result.regions = s2.regions.clone();
    }

    result
}

// ─── SpatialOracle ───────────────────────────────────────────────────────────

/// The spatial oracle evaluates spatial predicates against scene configurations
/// and detects transitions.
#[derive(Debug, Clone)]
pub struct SpatialOracle {
    /// The trajectory of scene configurations over time.
    trajectory: SpatialTrajectory,
    /// Registered spatial predicates to evaluate.
    predicates: Vec<(SpatialPredicateId, SpatialPredicate)>,
    /// Cache of previous valuation results.
    prev_valuations: HashMap<SpatialPredicateId, bool>,
    /// The AABB-based spatial index for the current scene.
    index: SpatialIndex,
    /// Next predicate ID.
    next_pred_id: u64,
}

impl SpatialOracle {
    pub fn new() -> Self {
        Self {
            trajectory: SpatialTrajectory::new(),
            predicates: Vec::new(),
            prev_valuations: HashMap::new(),
            index: SpatialIndex::new(),
            next_pred_id: 1,
        }
    }

    /// Create from an existing trajectory.
    pub fn from_trajectory(trajectory: SpatialTrajectory) -> Self {
        Self {
            trajectory,
            predicates: Vec::new(),
            prev_valuations: HashMap::new(),
            index: SpatialIndex::new(),
            next_pred_id: 1,
        }
    }

    /// Register a spatial predicate to be evaluated.
    pub fn register_predicate(&mut self, predicate: SpatialPredicate) -> SpatialPredicateId {
        let id = SpatialPredicateId(self.next_pred_id);
        self.next_pred_id += 1;
        self.predicates.push((id, predicate));
        id
    }

    /// Register a predicate with a specific ID.
    pub fn register_predicate_with_id(
        &mut self,
        id: SpatialPredicateId,
        predicate: SpatialPredicate,
    ) {
        self.predicates.push((id, predicate));
        if id.0 >= self.next_pred_id {
            self.next_pred_id = id.0 + 1;
        }
    }

    /// Add a scene snapshot at a given time.
    pub fn add_scene(&mut self, time: TimePoint, scene: SceneConfiguration) {
        self.trajectory.add_keyframe(time, scene);
    }

    /// Query the scene at a given time (interpolating if needed).
    pub fn query_scene_at(&self, time: TimePoint) -> Option<SceneConfiguration> {
        self.trajectory.interpolate_at(time)
    }

    /// Derive spatial fluents from a scene configuration.
    pub fn derive_spatial_fluents(
        &mut self,
        scene: &SceneConfiguration,
    ) -> Vec<(SpatialPredicateId, Fluent)> {
        self.index.build_from_scene(scene);

        let mut fluents = Vec::new();
        for (id, pred) in &self.predicates {
            let value = evaluate_spatial_predicate(pred, scene, &self.index);
            fluents.push((*id, Fluent::spatial(*id, value)));
        }
        fluents
    }

    /// Detect transitions between previous and current scene.
    pub fn detect_transitions(
        &mut self,
        scene: &SceneConfiguration,
        time: TimePoint,
    ) -> Vec<SpatialTransitionEvent> {
        self.index.build_from_scene(scene);

        let mut transitions = Vec::new();
        for (id, pred) in &self.predicates {
            let new_value = evaluate_spatial_predicate(pred, scene, &self.index);
            let old_value = self.prev_valuations.get(id).copied().unwrap_or(false);

            if old_value != new_value {
                transitions.push(SpatialTransitionEvent {
                    predicate_id: *id,
                    predicate: pred.clone(),
                    old_value,
                    new_value,
                    time,
                });
            }

            self.prev_valuations.insert(*id, new_value);
        }
        transitions
    }

    /// Evaluate all predicates at a given time and return valuations.
    pub fn evaluate_at(&mut self, time: TimePoint) -> Vec<PredicateValuation> {
        let scene = match self.query_scene_at(time) {
            Some(s) => s,
            None => return Vec::new(),
        };
        self.index.build_from_scene(&scene);

        let mut valuations = Vec::new();
        for (id, pred) in &self.predicates {
            let value = evaluate_spatial_predicate(pred, &scene, &self.index);
            valuations.push(PredicateValuation {
                predicate_id: *id,
                predicate: pred.clone(),
                value,
            });
        }
        valuations
    }

    /// Get the trajectory reference.
    pub fn trajectory(&self) -> &SpatialTrajectory {
        &self.trajectory
    }

    /// Get the mutable trajectory reference.
    pub fn trajectory_mut(&mut self) -> &mut SpatialTrajectory {
        &mut self.trajectory
    }

    /// Get the number of registered predicates.
    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    /// Reset the oracle's valuation cache.
    pub fn reset_cache(&mut self) {
        self.prev_valuations.clear();
    }

    /// Full scan: evaluate all predicates at uniform time steps and detect all transitions.
    pub fn scan_trajectory(
        &mut self,
        delta_t: Duration,
    ) -> Vec<SpatialTransitionEvent> {
        let samples = self.trajectory.sample_uniform(delta_t);
        let mut all_transitions = Vec::new();
        self.reset_cache();

        for (time, scene) in &samples {
            let transitions = self.detect_transitions(scene, *time);
            all_transitions.extend(transitions);
        }

        all_transitions
    }
}

impl Default for SpatialOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Predicate evaluation ────────────────────────────────────────────────────

/// Evaluate a single spatial predicate against a scene configuration.
fn evaluate_spatial_predicate(
    predicate: &SpatialPredicate,
    scene: &SceneConfiguration,
    _index: &SpatialIndex,
) -> bool {
    match predicate {
        SpatialPredicate::Near { a, b, distance } => {
            evaluate_near(scene, *a, *b, *distance)
        }
        SpatialPredicate::Inside { entity, region } => {
            evaluate_inside(scene, *entity, *region)
        }
        SpatialPredicate::Touching { a, b } => {
            evaluate_touching(scene, *a, *b)
        }
        SpatialPredicate::Overlapping { a, b } => {
            evaluate_overlapping(scene, *a, *b)
        }
        SpatialPredicate::GazingAt { gazer, target } => {
            evaluate_gazing_at(scene, *gazer, *target)
        }
        SpatialPredicate::Above { a, b } => {
            evaluate_above(scene, *a, *b)
        }
        SpatialPredicate::Facing { a, b, threshold } => {
            evaluate_facing(scene, *a, *b, *threshold)
        }
        SpatialPredicate::Custom { entities, .. } => {
            // Default: check if all entities exist in scene
            entities.iter().all(|e| scene.entities.contains_key(e))
        }
    }
}

fn evaluate_near(scene: &SceneConfiguration, a: EntityId, b: EntityId, distance: f64) -> bool {
    let ea = match scene.entities.get(&a) { Some(e) => e, None => return false };
    let eb = match scene.entities.get(&b) { Some(e) => e, None => return false };
    ea.transform.position.distance_to(&eb.transform.position) <= distance
}

fn evaluate_inside(scene: &SceneConfiguration, entity: EntityId, region: RegionId) -> bool {
    let e = match scene.entities.get(&entity) { Some(e) => e, None => return false };
    let r = match scene.regions.get(&region) { Some(r) => r, None => return false };
    r.contains(&e.transform.position)
}

fn evaluate_touching(scene: &SceneConfiguration, a: EntityId, b: EntityId) -> bool {
    let ea = match scene.entities.get(&a) { Some(e) => e, None => return false };
    let eb = match scene.entities.get(&b) { Some(e) => e, None => return false };
    ea.bounds.intersects(&eb.bounds)
}

fn evaluate_overlapping(scene: &SceneConfiguration, a: EntityId, b: EntityId) -> bool {
    evaluate_touching(scene, a, b)
}

fn evaluate_gazing_at(scene: &SceneConfiguration, gazer: EntityId, target: EntityId) -> bool {
    let eg = match scene.entities.get(&gazer) { Some(e) => e, None => return false };
    let et = match scene.entities.get(&target) { Some(e) => e, None => return false };

    let to_target = Vector3::new(
        et.transform.position.x - eg.transform.position.x,
        et.transform.position.y - eg.transform.position.y,
        et.transform.position.z - eg.transform.position.z,
    );
    let dist = to_target.magnitude();
    if dist < 1e-9 {
        return true;
    }
    let dir = to_target.normalized();
    let forward = eg.forward.normalized();
    let dot = forward.dot(&dir);
    dot > 0.9 // ~25° cone
}

fn evaluate_above(scene: &SceneConfiguration, a: EntityId, b: EntityId) -> bool {
    let ea = match scene.entities.get(&a) { Some(e) => e, None => return false };
    let eb = match scene.entities.get(&b) { Some(e) => e, None => return false };
    ea.transform.position.y > eb.transform.position.y
}

fn evaluate_facing(
    scene: &SceneConfiguration,
    a: EntityId,
    b: EntityId,
    threshold: f64,
) -> bool {
    let ea = match scene.entities.get(&a) { Some(e) => e, None => return false };
    let eb = match scene.entities.get(&b) { Some(e) => e, None => return false };
    let to_b = Vector3::new(
        eb.transform.position.x - ea.transform.position.x,
        eb.transform.position.y - ea.transform.position.y,
        eb.transform.position.z - ea.transform.position.z,
    );
    let dist = to_b.magnitude();
    if dist < 1e-9 {
        return true;
    }
    let dir = to_b.normalized();
    let forward = ea.forward.normalized();
    forward.dot(&dir) > threshold
}

// ─── Lipschitz sampling bound ────────────────────────────────────────────────

/// Compute the minimum sampling interval that guarantees no spatial predicate
/// transitions are missed, given a Lipschitz constant on entity velocity.
///
/// The Lipschitz constant `L` bounds the maximum speed: `|x(t2) - x(t1)| ≤ L * |t2 - t1|`.
/// For a predicate with threshold `d`, the sampling interval must satisfy:
/// `delta_t ≤ d / L` to guarantee that all crossings are detected.
pub fn compute_sampling_bound(lipschitz_constant: f64, threshold: f64) -> Duration {
    if lipschitz_constant <= 0.0 {
        return Duration::from_secs(f64::MAX);
    }
    Duration::from_secs(threshold / lipschitz_constant)
}

/// Verify that a trajectory is adequately sampled for a given delta_t.
///
/// Returns true if consecutive keyframes are at most `delta_t` apart.
pub fn verify_sampling_adequacy(trajectory: &SpatialTrajectory, delta_t: Duration) -> bool {
    let times = trajectory.keyframe_times();
    if times.len() < 2 {
        return true;
    }
    for pair in times.windows(2) {
        let gap = pair[1].0 - pair[0].0;
        if gap > delta_t.0 * 1.001 {
            return false;
        }
    }
    true
}

/// Estimate the Lipschitz constant from a trajectory by computing max velocity.
pub fn estimate_lipschitz_constant(trajectory: &SpatialTrajectory) -> f64 {
    let times = trajectory.keyframe_times();
    if times.len() < 2 {
        return 0.0;
    }

    let mut max_speed = 0.0_f64;

    for pair in times.windows(2) {
        let s1 = trajectory.keyframe_at(pair[0]).unwrap();
        let s2 = trajectory.keyframe_at(pair[1]).unwrap();
        let dt = pair[1].0 - pair[0].0;
        if dt < 1e-12 {
            continue;
        }

        for (id, e1) in &s1.entities {
            if let Some(e2) = s2.entities.get(id) {
                let dist = e1.transform.position.distance_to(&e2.transform.position);
                let speed = dist / dt;
                max_speed = max_speed.max(speed);
            }
        }
    }

    max_speed
}

/// Find times at which a specific spatial predicate transitions, using adaptive sampling.
pub fn find_transition_times(
    oracle: &mut SpatialOracle,
    predicate_id: SpatialPredicateId,
    initial_delta_t: Duration,
    refinement_iterations: usize,
) -> Vec<TimePoint> {
    let _span = match oracle.trajectory().time_span() {
        Some(s) => s,
        None => return Vec::new(),
    };

    // Coarse scan
    let mut transitions: Vec<TimePoint> = Vec::new();
    let coarse = oracle.scan_trajectory(initial_delta_t);
    for t in &coarse {
        if t.predicate_id == predicate_id {
            transitions.push(t.time);
        }
    }

    // Refine each transition time via bisection
    for _ in 0..refinement_iterations {
        let mut refined = Vec::new();
        for &t in &transitions {
            let low = TimePoint(t.0 - initial_delta_t.0);
            let high = t;
            let mid = TimePoint((low.0 + high.0) / 2.0);

            oracle.reset_cache();
            // Evaluate at surrounding points
            let val_low = oracle.evaluate_at(low);
            let val_mid = oracle.evaluate_at(mid);

            let pred_low = val_low
                .iter()
                .find(|v| v.predicate_id == predicate_id)
                .map_or(false, |v| v.value);
            let pred_mid = val_mid
                .iter()
                .find(|v| v.predicate_id == predicate_id)
                .map_or(false, |v| v.value);

            if pred_low != pred_mid {
                refined.push(mid);
            } else {
                refined.push(t);
            }
        }
        transitions = refined;
    }

    transitions
}

// ─── Scene snapshot management ───────────────────────────────────────────────

/// Manages a sequence of scene snapshots with efficient storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSnapshotManager {
    snapshots: BTreeMap<OrderedFloat<f64>, SceneConfiguration>,
    max_snapshots: usize,
}

impl SceneSnapshotManager {
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: BTreeMap::new(),
            max_snapshots,
        }
    }

    /// Store a snapshot, evicting the oldest if at capacity.
    pub fn store(&mut self, time: TimePoint, scene: SceneConfiguration) {
        if self.snapshots.len() >= self.max_snapshots {
            if let Some(oldest) = self.snapshots.keys().next().copied() {
                self.snapshots.remove(&oldest);
            }
        }
        self.snapshots.insert(OrderedFloat(time.0), scene);
    }

    /// Get a snapshot at the given time.
    pub fn get(&self, time: TimePoint) -> Option<&SceneConfiguration> {
        self.snapshots.get(&OrderedFloat(time.0))
    }

    /// Get the most recent snapshot at or before a time.
    pub fn get_before(&self, time: TimePoint) -> Option<(TimePoint, &SceneConfiguration)> {
        self.snapshots
            .range(..=OrderedFloat(time.0))
            .next_back()
            .map(|(t, s)| (TimePoint(t.0), s))
    }

    /// Get the count of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Get the time span of stored snapshots.
    pub fn time_span(&self) -> Option<TimeInterval> {
        let first = self.snapshots.keys().next()?;
        let last = self.snapshots.keys().next_back()?;
        Some(TimeInterval::new(TimePoint(first.0), TimePoint(last.0)))
    }

    /// Convert to a SpatialTrajectory.
    pub fn to_trajectory(&self) -> SpatialTrajectory {
        let mut traj = SpatialTrajectory::new();
        for (t, s) in &self.snapshots {
            traj.add_keyframe(TimePoint(t.0), s.clone());
        }
        traj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_scene(x1: f64, x2: f64) -> SceneConfiguration {
        let mut scene = SceneConfiguration::new();
        let mut e1 = SceneEntity::new(
            EntityId(1),
            "hand",
            Transform3D::from_position(Point3::new(x1, 0.0, 0.0)),
        );
        e1.forward = Vector3::new(1.0, 0.0, 0.0);
        let mut e2 = SceneEntity::new(
            EntityId(2),
            "object",
            Transform3D::from_position(Point3::new(x2, 0.0, 0.0)),
        );
        e2.forward = Vector3::new(-1.0, 0.0, 0.0);
        scene.add_entity(e1);
        scene.add_entity(e2);
        scene.add_region(
            RegionId(1),
            SpatialRegion::Sphere {
                center: Point3::origin(),
                radius: 5.0,
            },
        );
        scene
    }

    #[test]
    fn test_near_predicate() {
        let scene = make_test_scene(0.0, 1.0);
        let index = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene);
            idx
        };
        let pred = SpatialPredicate::Near {
            a: EntityId(1),
            b: EntityId(2),
            distance: 2.0,
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &index));

        let scene_far = make_test_scene(0.0, 10.0);
        let idx2 = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene_far);
            idx
        };
        assert!(!evaluate_spatial_predicate(&pred, &scene_far, &idx2));
    }

    #[test]
    fn test_inside_predicate() {
        let scene = make_test_scene(0.0, 1.0);
        let index = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene);
            idx
        };
        let pred = SpatialPredicate::Inside {
            entity: EntityId(1),
            region: RegionId(1),
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &index));
    }

    #[test]
    fn test_touching_predicate() {
        let scene = make_test_scene(0.0, 0.5);
        let index = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene);
            idx
        };
        let pred = SpatialPredicate::Touching {
            a: EntityId(1),
            b: EntityId(2),
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &index));

        let scene_far = make_test_scene(0.0, 10.0);
        let idx2 = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene_far);
            idx
        };
        assert!(!evaluate_spatial_predicate(&pred, &scene_far, &idx2));
    }

    #[test]
    fn test_gazing_at_predicate() {
        let scene = make_test_scene(0.0, 2.0);
        let index = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene);
            idx
        };
        let pred = SpatialPredicate::GazingAt {
            gazer: EntityId(1),
            target: EntityId(2),
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &index));
    }

    #[test]
    fn test_above_predicate() {
        let mut scene = SceneConfiguration::new();
        scene.add_entity(SceneEntity::new(
            EntityId(1), "top",
            Transform3D::from_position(Point3::new(0.0, 5.0, 0.0)),
        ));
        scene.add_entity(SceneEntity::new(
            EntityId(2), "bottom",
            Transform3D::from_position(Point3::new(0.0, 0.0, 0.0)),
        ));
        let index = {
            let mut idx = SpatialIndex::new();
            idx.build_from_scene(&scene);
            idx
        };
        let pred = SpatialPredicate::Above {
            a: EntityId(1),
            b: EntityId(2),
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &index));
    }

    #[test]
    fn test_spatial_oracle_derive_fluents() {
        let mut oracle = SpatialOracle::new();
        let _pred_id = oracle.register_predicate(SpatialPredicate::Near {
            a: EntityId(1),
            b: EntityId(2),
            distance: 2.0,
        });

        let scene = make_test_scene(0.0, 1.0);
        let fluents = oracle.derive_spatial_fluents(&scene);
        assert_eq!(fluents.len(), 1);
        assert!(fluents[0].1.holds());
    }

    #[test]
    fn test_spatial_oracle_detect_transitions() {
        let mut oracle = SpatialOracle::new();
        oracle.register_predicate(SpatialPredicate::Near {
            a: EntityId(1),
            b: EntityId(2),
            distance: 2.0,
        });

        let scene1 = make_test_scene(0.0, 1.0); // near
        let scene2 = make_test_scene(0.0, 10.0); // far

        let t1 = oracle.detect_transitions(&scene1, TimePoint::from_secs(0.0));
        // First evaluation: old was false (default), new is true
        assert_eq!(t1.len(), 1);
        assert!(t1[0].is_onset());

        let t2 = oracle.detect_transitions(&scene2, TimePoint::from_secs(1.0));
        assert_eq!(t2.len(), 1);
        assert!(t2[0].is_offset());
    }

    #[test]
    fn test_spatial_trajectory_interpolation() {
        let mut traj = SpatialTrajectory::new();
        traj.add_keyframe(TimePoint::from_secs(0.0), make_test_scene(0.0, 10.0));
        traj.add_keyframe(TimePoint::from_secs(1.0), make_test_scene(10.0, 10.0));

        let mid = traj.interpolate_at(TimePoint::from_secs(0.5)).unwrap();
        let e1 = mid.entities.get(&EntityId(1)).unwrap();
        assert!((e1.transform.position.x - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_sampling_bound() {
        let bound = compute_sampling_bound(2.0, 1.0);
        assert!((bound.as_secs() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_sampling_adequacy() {
        let mut traj = SpatialTrajectory::new();
        traj.add_keyframe(TimePoint::from_secs(0.0), SceneConfiguration::new());
        traj.add_keyframe(TimePoint::from_secs(0.1), SceneConfiguration::new());
        traj.add_keyframe(TimePoint::from_secs(0.2), SceneConfiguration::new());

        assert!(verify_sampling_adequacy(&traj, Duration::from_secs(0.1)));
        assert!(!verify_sampling_adequacy(&traj, Duration::from_secs(0.05)));
    }

    #[test]
    fn test_estimate_lipschitz() {
        let mut traj = SpatialTrajectory::new();
        traj.add_keyframe(TimePoint::from_secs(0.0), make_test_scene(0.0, 10.0));
        traj.add_keyframe(TimePoint::from_secs(1.0), make_test_scene(5.0, 10.0));

        let lip = estimate_lipschitz_constant(&traj);
        assert!((lip - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_scene_snapshot_manager() {
        let mut mgr = SceneSnapshotManager::new(3);
        mgr.store(TimePoint::from_secs(0.0), make_test_scene(0.0, 1.0));
        mgr.store(TimePoint::from_secs(1.0), make_test_scene(1.0, 2.0));
        mgr.store(TimePoint::from_secs(2.0), make_test_scene(2.0, 3.0));
        assert_eq!(mgr.len(), 3);

        mgr.store(TimePoint::from_secs(3.0), make_test_scene(3.0, 4.0));
        assert_eq!(mgr.len(), 3);
        assert!(mgr.get(TimePoint::from_secs(0.0)).is_none());
    }

    #[test]
    fn test_spatial_oracle_scan_trajectory() {
        let mut oracle = SpatialOracle::new();
        oracle.register_predicate(SpatialPredicate::Near {
            a: EntityId(1),
            b: EntityId(2),
            distance: 2.0,
        });
        oracle.add_scene(TimePoint::from_secs(0.0), make_test_scene(0.0, 1.0));
        oracle.add_scene(TimePoint::from_secs(1.0), make_test_scene(0.0, 10.0));
        oracle.add_scene(TimePoint::from_secs(2.0), make_test_scene(0.0, 1.0));

        let transitions = oracle.scan_trajectory(Duration::from_secs(0.5));
        assert!(transitions.len() >= 2); // onset, offset, onset
    }

    #[test]
    fn test_trajectory_time_span() {
        let mut traj = SpatialTrajectory::new();
        traj.add_keyframe(TimePoint::from_secs(1.0), SceneConfiguration::new());
        traj.add_keyframe(TimePoint::from_secs(5.0), SceneConfiguration::new());
        let span = traj.time_span().unwrap();
        assert!((span.start.0 - 1.0).abs() < 1e-9);
        assert!((span.end.0 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_spatial_transition_event_to_event() {
        let st = SpatialTransitionEvent {
            predicate_id: SpatialPredicateId(1),
            predicate: SpatialPredicate::Near {
                a: EntityId(1),
                b: EntityId(2),
                distance: 2.0,
            },
            old_value: false,
            new_value: true,
            time: TimePoint::from_secs(1.0),
        };
        let event = st.to_event(EventId(42));
        assert_eq!(event.id, EventId(42));
        assert_eq!(event.time, TimePoint::from_secs(1.0));
    }

    #[test]
    fn test_facing_predicate() {
        let mut scene = SceneConfiguration::new();
        let mut e1 = SceneEntity::new(
            EntityId(1), "a",
            Transform3D::from_position(Point3::new(0.0, 0.0, 0.0)),
        );
        e1.forward = Vector3::new(1.0, 0.0, 0.0);
        let e2 = SceneEntity::new(
            EntityId(2), "b",
            Transform3D::from_position(Point3::new(5.0, 0.0, 0.0)),
        );
        scene.add_entity(e1);
        scene.add_entity(e2);

        let idx = {
            let mut i = SpatialIndex::new();
            i.build_from_scene(&scene);
            i
        };
        let pred = SpatialPredicate::Facing {
            a: EntityId(1),
            b: EntityId(2),
            threshold: 0.9,
        };
        assert!(evaluate_spatial_predicate(&pred, &scene, &idx));
    }
}
