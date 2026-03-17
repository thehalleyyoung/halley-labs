//! Spatial predicate evaluation for scene configurations.
//!
//! Evaluates spatial predicates (proximity, containment, gaze, intersection,
//! touching) against concrete scene configurations, supports incremental
//! evaluation, dependency tracking, and geometric consistency checking.

use std::collections::{HashMap, HashSet};

use choreo_types::geometry::{AABB, Point3, Sphere, Vector3};
use choreo_types::spatial::{
    EntityId, PredicateValuation, RegionId, SceneConfiguration, SceneEntity,
    SpatialPredicate, SpatialPredicateEvaluator, SpatialPredicateId, SpatialRegion,
};

// GJK collision detection can be used for precise predicate evaluation.
#[allow(unused_imports)]
use crate::gjk::{gjk_distance, gjk_intersection, CollisionShape, GjkResult};

// Re-export GJK types used by this module.
#[allow(unused_imports)]
use crate::gjk;

// ─── evaluator ───────────────────────────────────────────────────────────────

/// Concrete implementation of the spatial predicate evaluator.
#[derive(Debug, Clone)]
pub struct SpatialPredicateEvaluatorImpl {
    /// Cached previous valuation for incremental evaluation.
    cached_valuation: Option<PredicateValuation>,
    /// Entity versions for change detection.
    entity_versions: HashMap<EntityId, u64>,
    /// Region versions for change detection.
    region_versions: HashMap<RegionId, u64>,
    /// Global version counter.
    version_counter: u64,
}

impl SpatialPredicateEvaluatorImpl {
    pub fn new() -> Self {
        Self {
            cached_valuation: None,
            entity_versions: HashMap::new(),
            region_versions: HashMap::new(),
            version_counter: 0,
        }
    }

    /// Evaluate a single predicate.
    pub fn evaluate_predicate(
        &self,
        predicate: &SpatialPredicate,
        scene: &SceneConfiguration,
    ) -> bool {
        match predicate {
            SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                threshold,
            } => self.evaluate_proximity(entity_a, entity_b, *threshold, scene),
            SpatialPredicate::Inside { entity, region } => {
                self.evaluate_inside(entity, region, scene)
            }
            SpatialPredicate::GazeAt {
                source,
                target,
                cone_angle,
            } => self.evaluate_gaze_at(source, target, *cone_angle, scene),
            SpatialPredicate::Containment { inner, outer } => {
                self.evaluate_containment(inner, outer, scene)
            }
            SpatialPredicate::Intersection {
                region_a,
                region_b,
            } => self.evaluate_intersection(region_a, region_b, scene),
            SpatialPredicate::Touching {
                entity_a,
                entity_b,
            } => self.evaluate_touching(entity_a, entity_b, scene),
        }
    }

    /// Evaluate proximity: are two entities within `threshold` distance?
    pub fn evaluate_proximity(
        &self,
        entity_a: &EntityId,
        entity_b: &EntityId,
        threshold: f64,
        scene: &SceneConfiguration,
    ) -> bool {
        let (Some(ea), Some(eb)) = (scene.entities.get(entity_a), scene.entities.get(entity_b))
        else {
            return false;
        };

        let dist = entity_distance(ea, eb);
        dist <= threshold
    }

    /// Evaluate inside: is an entity inside a region?
    pub fn evaluate_inside(
        &self,
        entity: &EntityId,
        region: &RegionId,
        scene: &SceneConfiguration,
    ) -> bool {
        let Some(ent) = scene.entities.get(entity) else {
            return false;
        };
        let Some(reg) = scene.regions.get(region) else {
            return false;
        };
        point_in_region(&ent.transform.position_point(), reg)
    }

    /// Evaluate gaze: is `source` looking at `target` within `cone_angle` radians?
    pub fn evaluate_gaze_at(
        &self,
        source: &EntityId,
        target: &EntityId,
        cone_angle: f64,
        scene: &SceneConfiguration,
    ) -> bool {
        let (Some(src), Some(tgt)) = (scene.entities.get(source), scene.entities.get(target))
        else {
            return false;
        };

        let source_pos = src.transform.position_point();
        let target_pos = tgt.transform.position_point();

        // Gaze direction: forward of the source entity (local +Z after rotation).
        let rot = src.transform.quaternion();
        let gaze_dir = rot * Vector3::new(0.0, 0.0, -1.0); // forward = -Z convention
        let gaze_dir = gaze_dir.normalize();

        let to_target = target_pos - source_pos;
        let dist = to_target.norm();
        if dist < 1e-10 {
            return true; // On top of target.
        }
        let to_target_norm = to_target / dist;

        let cos_angle = gaze_dir.dot(&to_target_norm);
        let half_cone = cone_angle / 2.0;
        cos_angle >= half_cone.cos()
    }

    /// Evaluate gaze cone intersection with a sphere (for target bounding sphere).
    pub fn evaluate_gaze_cone_sphere(
        &self,
        source_pos: &Point3,
        gaze_dir: &Vector3,
        cone_angle: f64,
        sphere: &Sphere,
    ) -> bool {
        let to_center = sphere.center_point() - source_pos;
        let dist = to_center.norm();
        if dist < 1e-10 {
            return true;
        }
        let to_center_norm = to_center / dist;
        let cos_angle = gaze_dir.dot(&to_center_norm);
        let half_cone = cone_angle / 2.0;

        // Cone-sphere intersection: account for sphere radius.
        let angular_extent = (sphere.radius / dist).asin();
        cos_angle >= (half_cone + angular_extent).cos()
    }

    /// Evaluate gaze cone intersection with an AABB.
    pub fn evaluate_gaze_cone_aabb(
        &self,
        source_pos: &Point3,
        gaze_dir: &Vector3,
        cone_angle: f64,
        aabb: &AABB,
    ) -> bool {
        // Conservative test: convert AABB to bounding sphere.
        let center = aabb.center();
        let half_ext = aabb.half_extents();
        let radius = half_ext.norm();
        let sphere = Sphere::new([center.x, center.y, center.z], radius);
        self.evaluate_gaze_cone_sphere(source_pos, gaze_dir, cone_angle, &sphere)
    }

    /// Evaluate containment: is `inner` region fully inside `outer` region?
    pub fn evaluate_containment(
        &self,
        inner: &RegionId,
        outer: &RegionId,
        scene: &SceneConfiguration,
    ) -> bool {
        let (Some(inner_reg), Some(outer_reg)) =
            (scene.regions.get(inner), scene.regions.get(outer))
        else {
            return false;
        };
        region_contains_region(outer_reg, inner_reg)
    }

    /// Evaluate intersection: do two regions overlap?
    pub fn evaluate_intersection(
        &self,
        region_a: &RegionId,
        region_b: &RegionId,
        scene: &SceneConfiguration,
    ) -> bool {
        let (Some(ra), Some(rb)) = (scene.regions.get(region_a), scene.regions.get(region_b))
        else {
            return false;
        };
        regions_intersect(ra, rb)
    }

    /// Evaluate touching: are two entities in contact (distance ≈ 0)?
    pub fn evaluate_touching(
        &self,
        entity_a: &EntityId,
        entity_b: &EntityId,
        scene: &SceneConfiguration,
    ) -> bool {
        let (Some(ea), Some(eb)) = (scene.entities.get(entity_a), scene.entities.get(entity_b))
        else {
            return false;
        };

        let dist = entity_distance(ea, eb);
        dist <= 0.01 // touching threshold
    }

    /// Batch evaluate all predicates for a scene configuration.
    pub fn batch_evaluate(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        scene: &SceneConfiguration,
    ) -> PredicateValuation {
        let mut valuation = PredicateValuation::new();
        for (id, pred) in predicates {
            let value = self.evaluate_predicate(pred, scene);
            valuation.set(id.clone(), value);
        }
        valuation
    }

    /// Incremental evaluation: only re-evaluate predicates whose inputs changed.
    pub fn incremental_evaluate(
        &mut self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        scene: &SceneConfiguration,
        changed_entities: &HashSet<EntityId>,
        changed_regions: &HashSet<RegionId>,
    ) -> PredicateValuation {
        let prev = self.cached_valuation.take().unwrap_or_default();
        let mut valuation = PredicateValuation::new();

        for (id, pred) in predicates {
            let needs_eval = predicate_depends_on_changed(pred, changed_entities, changed_regions);

            if needs_eval {
                let value = self.evaluate_predicate(pred, scene);
                valuation.set(id.clone(), value);
            } else if let Some(cached_value) = prev.get(id) {
                valuation.set(id.clone(), cached_value);
            } else {
                let value = self.evaluate_predicate(pred, scene);
                valuation.set(id.clone(), value);
            }
        }

        self.cached_valuation = Some(valuation.clone());
        self.version_counter += 1;
        valuation
    }

    /// Mark an entity as changed.
    pub fn mark_entity_changed(&mut self, entity: &EntityId) {
        self.version_counter += 1;
        self.entity_versions
            .insert(entity.clone(), self.version_counter);
    }

    /// Mark a region as changed.
    pub fn mark_region_changed(&mut self, region: &RegionId) {
        self.version_counter += 1;
        self.region_versions
            .insert(region.clone(), self.version_counter);
    }
}

impl Default for SpatialPredicateEvaluatorImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialPredicateEvaluator for SpatialPredicateEvaluatorImpl {
    fn evaluate(&self, predicate: &SpatialPredicate, scene: &SceneConfiguration) -> bool {
        self.evaluate_predicate(predicate, scene)
    }

    fn evaluate_all(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        scene: &SceneConfiguration,
    ) -> PredicateValuation {
        self.batch_evaluate(predicates, scene)
    }
}

// ─── dependency tracking ─────────────────────────────────────────────────────

/// Track which entities and regions a predicate depends on.
#[derive(Debug, Clone)]
pub struct PredicateDependencies {
    pub entities: HashSet<EntityId>,
    pub regions: HashSet<RegionId>,
}

impl PredicateDependencies {
    pub fn from_predicate(pred: &SpatialPredicate) -> Self {
        let mut entities = HashSet::new();
        let mut regions = HashSet::new();

        for e in pred.involved_entities() {
            entities.insert(e.clone());
        }
        for r in pred.involved_regions() {
            regions.insert(r.clone());
        }

        Self { entities, regions }
    }
}

fn predicate_depends_on_changed(
    pred: &SpatialPredicate,
    changed_entities: &HashSet<EntityId>,
    changed_regions: &HashSet<RegionId>,
) -> bool {
    for e in pred.involved_entities() {
        if changed_entities.contains(e) {
            return true;
        }
    }
    for r in pred.involved_regions() {
        if changed_regions.contains(r) {
            return true;
        }
    }
    false
}

// ─── geometric consistency checker ──────────────────────────────────────────

/// Verifies that predicate valuations are geometrically realizable.
#[derive(Debug)]
pub struct GeometricConsistencyChecker;

impl GeometricConsistencyChecker {
    pub fn new() -> Self {
        Self
    }

    /// Check monotonicity: Proximity(a,b,r1) => Proximity(a,b,r2) for r1 <= r2.
    pub fn check_monotonicity(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        valuation: &PredicateValuation,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        // Group proximity predicates by entity pair.
        let mut proximity_groups: HashMap<(EntityId, EntityId), Vec<(SpatialPredicateId, f64)>> =
            HashMap::new();

        for (id, pred) in predicates {
            if let SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                threshold,
            } = pred
            {
                let key = normalize_entity_pair(entity_a, entity_b);
                proximity_groups
                    .entry(key)
                    .or_default()
                    .push((id.clone(), *threshold));
            }
        }

        for (pair, mut group) in proximity_groups {
            group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    let val_i = valuation.get(&group[i].0).unwrap_or(false);
                    let val_j = valuation.get(&group[j].0).unwrap_or(false);
                    // If closer threshold is true, farther threshold must also be true.
                    if val_i && !val_j {
                        violations.push(ConsistencyViolation::MonotonicityViolation {
                            entity_a: pair.0.clone(),
                            entity_b: pair.1.clone(),
                            smaller_threshold: group[i].1,
                            larger_threshold: group[j].1,
                        });
                    }
                }
            }
        }

        violations
    }

    /// Check triangle inequality: dist(a,c) <= dist(a,b) + dist(b,c).
    pub fn check_triangle_inequality(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        valuation: &PredicateValuation,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        // Build distance bounds from proximity predicates.
        let mut distance_bounds: HashMap<(EntityId, EntityId), (f64, bool)> = HashMap::new();

        for (id, pred) in predicates {
            if let SpatialPredicate::Proximity {
                entity_a,
                entity_b,
                threshold,
            } = pred
            {
                let key = normalize_entity_pair(entity_a, entity_b);
                let val = valuation.get(id).unwrap_or(false);
                distance_bounds.insert(key, (*threshold, val));
            }
        }

        // Check all triples.
        let entities: Vec<EntityId> = distance_bounds
            .keys()
            .flat_map(|(a, b)| vec![a.clone(), b.clone()])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                for k in (j + 1)..entities.len() {
                    let ab = normalize_entity_pair(&entities[i], &entities[j]);
                    let bc = normalize_entity_pair(&entities[j], &entities[k]);
                    let ac = normalize_entity_pair(&entities[i], &entities[k]);

                    // If Proximity(a,b,r1)=true and Proximity(b,c,r2)=true,
                    // then we know dist(a,b)<=r1 and dist(b,c)<=r2,
                    // so dist(a,c)<=r1+r2 by triangle inequality.
                    // If Proximity(a,c,r3)=false and r3 >= r1+r2, that's a violation.
                    if let (Some(&(r_ab, true)), Some(&(r_bc, true))) =
                        (distance_bounds.get(&ab), distance_bounds.get(&bc))
                    {
                        if let Some(&(r_ac, false)) = distance_bounds.get(&ac) {
                            if r_ac >= r_ab + r_bc {
                                violations.push(
                                    ConsistencyViolation::TriangleInequalityViolation {
                                        entity_a: entities[i].clone(),
                                        entity_b: entities[j].clone(),
                                        entity_c: entities[k].clone(),
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    /// Check containment consistency: if A ⊂ B and B ⊂ C, then A ⊂ C.
    pub fn check_containment_consistency(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        valuation: &PredicateValuation,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        // Build containment graph.
        let mut containment_edges: Vec<(RegionId, RegionId)> = Vec::new(); // (inner, outer)

        for (id, pred) in predicates {
            if let SpatialPredicate::Containment { inner, outer } = pred {
                if valuation.get(id).unwrap_or(false) {
                    containment_edges.push((inner.clone(), outer.clone()));
                }
            }
        }

        // Check transitivity.
        for (a_inner, a_outer) in &containment_edges {
            for (b_inner, b_outer) in &containment_edges {
                if a_outer == b_inner {
                    // a_inner ⊂ a_outer = b_inner ⊂ b_outer
                    // So a_inner ⊂ b_outer should hold.
                    let transitive_holds = containment_edges
                        .iter()
                        .any(|(i, o)| i == a_inner && o == b_outer);

                    // Also check if there's an explicit predicate for this.
                    let has_predicate = predicates.iter().any(|(_pid, p)| {
                        if let SpatialPredicate::Containment { inner, outer } = p {
                            inner == a_inner && outer == b_outer
                        } else {
                            false
                        }
                    });

                    if has_predicate && !transitive_holds {
                        // There is a predicate for this pair but it's false.
                        let pred_value = predicates
                            .iter()
                            .find_map(|(pid, p)| {
                                if let SpatialPredicate::Containment { inner, outer } = p {
                                    if inner == a_inner && outer == b_outer {
                                        return valuation.get(pid);
                                    }
                                }
                                None
                            })
                            .unwrap_or(false);

                        if !pred_value {
                            violations.push(
                                ConsistencyViolation::ContainmentTransitivityViolation {
                                    inner: a_inner.clone(),
                                    middle: a_outer.clone(),
                                    outer: b_outer.clone(),
                                },
                            );
                        }
                    }
                }
            }
        }

        violations
    }

    /// Run all consistency checks.
    pub fn check_all(
        &self,
        predicates: &[(SpatialPredicateId, SpatialPredicate)],
        valuation: &PredicateValuation,
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();
        violations.extend(self.check_monotonicity(predicates, valuation));
        violations.extend(self.check_triangle_inequality(predicates, valuation));
        violations.extend(self.check_containment_consistency(predicates, valuation));
        violations
    }
}

impl Default for GeometricConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// A consistency violation found by the checker.
#[derive(Debug, Clone)]
pub enum ConsistencyViolation {
    MonotonicityViolation {
        entity_a: EntityId,
        entity_b: EntityId,
        smaller_threshold: f64,
        larger_threshold: f64,
    },
    TriangleInequalityViolation {
        entity_a: EntityId,
        entity_b: EntityId,
        entity_c: EntityId,
    },
    ContainmentTransitivityViolation {
        inner: RegionId,
        middle: RegionId,
        outer: RegionId,
    },
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn normalize_entity_pair(a: &EntityId, b: &EntityId) -> (EntityId, EntityId) {
    if a.0 <= b.0 {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    }
}

fn entity_distance(a: &SceneEntity, b: &SceneEntity) -> f64 {
    let pa = a.transform.position_point();
    let pb = b.transform.position_point();

    // Use AABB-based distance for better accuracy.
    let aabb_a = translate_aabb(&a.bounds, &pa);
    let aabb_b = translate_aabb(&b.bounds, &pb);

    aabb_distance(&aabb_a, &aabb_b)
}

fn translate_aabb(aabb: &AABB, offset: &Point3) -> AABB {
    AABB::new(
        [
            aabb.min[0] + offset.x,
            aabb.min[1] + offset.y,
            aabb.min[2] + offset.z,
        ],
        [
            aabb.max[0] + offset.x,
            aabb.max[1] + offset.y,
            aabb.max[2] + offset.z,
        ],
    )
}

fn aabb_distance(a: &AABB, b: &AABB) -> f64 {
    let mut dist_sq = 0.0;
    for i in 0..3 {
        let gap = (b.min[i] - a.max[i]).max(a.min[i] - b.max[i]).max(0.0);
        dist_sq += gap * gap;
    }
    dist_sq.sqrt()
}

fn point_in_region(point: &Point3, region: &SpatialRegion) -> bool {
    match region {
        SpatialRegion::Aabb(aabb) => aabb.contains_point(point),
        SpatialRegion::Sphere { center, radius } => {
            let dx = point.x - center[0];
            let dy = point.y - center[1];
            let dz = point.z - center[2];
            dx * dx + dy * dy + dz * dz <= radius * radius
        }
        SpatialRegion::ConvexHull { points } => point_in_convex_hull(point, points),
        SpatialRegion::Composite { regions } => regions.iter().any(|r| point_in_region(point, r)),
    }
}

fn point_in_convex_hull(point: &Point3, hull_points: &[[f64; 3]]) -> bool {
    if hull_points.len() < 4 {
        return false;
    }
    // Approximate: check if inside AABB of hull points.
    let pts: Vec<Point3> = hull_points
        .iter()
        .map(|p| Point3::new(p[0], p[1], p[2]))
        .collect();
    let aabb = AABB::from_points(&pts);
    aabb.contains_point(point)
}

fn region_contains_region(outer: &SpatialRegion, inner: &SpatialRegion) -> bool {
    let outer_aabb = outer.to_aabb();
    let inner_aabb = inner.to_aabb();
    outer_aabb.contains_aabb(&inner_aabb)
}

fn regions_intersect(a: &SpatialRegion, b: &SpatialRegion) -> bool {
    let aabb_a = a.to_aabb();
    let aabb_b = b.to_aabb();

    if !aabb_a.intersects(&aabb_b) {
        return false;
    }

    // For spheres, do precise check.
    match (a, b) {
        (
            SpatialRegion::Sphere {
                center: ca,
                radius: ra,
            },
            SpatialRegion::Sphere {
                center: cb,
                radius: rb,
            },
        ) => {
            let dx = ca[0] - cb[0];
            let dy = ca[1] - cb[1];
            let dz = ca[2] - cb[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let r_sum = ra + rb;
            dist_sq <= r_sum * r_sum
        }
        _ => {
            // Fall back to AABB intersection.
            aabb_a.intersects(&aabb_b)
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_types::geometry::Transform3D;

    fn make_entity(id: &str, x: f64, y: f64, z: f64) -> (EntityId, SceneEntity) {
        let eid = EntityId(id.to_string());
        let ent = SceneEntity {
            id: eid.clone(),
            transform: Transform3D::from_position(x, y, z),
            bounds: AABB::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            region: None,
        };
        (eid, ent)
    }

    fn make_scene(entities: Vec<(EntityId, SceneEntity)>) -> SceneConfiguration {
        let mut scene = SceneConfiguration::new();
        for (id, ent) in entities {
            scene.entities.insert(id, ent);
        }
        scene
    }

    #[test]
    fn test_proximity_true() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 1.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_proximity(&a_id, &b_id, 2.0, &scene));
    }

    #[test]
    fn test_proximity_false() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 10.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(!eval.evaluate_proximity(&a_id, &b_id, 2.0, &scene));
    }

    #[test]
    fn test_inside_aabb_region() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let region_id = RegionId("room".to_string());
        let region = SpatialRegion::Aabb(AABB::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]));

        let mut scene = make_scene(vec![(a_id.clone(), a_ent)]);
        scene.regions.insert(region_id.clone(), region);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_inside(&a_id, &region_id, &scene));
    }

    #[test]
    fn test_inside_sphere_region() {
        let (a_id, a_ent) = make_entity("a", 1.0, 0.0, 0.0);
        let region_id = RegionId("zone".to_string());
        let region = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };

        let mut scene = make_scene(vec![(a_id.clone(), a_ent)]);
        scene.regions.insert(region_id.clone(), region);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_inside(&a_id, &region_id, &scene));
    }

    #[test]
    fn test_touching() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 1.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_touching(&a_id, &b_id, &scene));
    }

    #[test]
    fn test_not_touching() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 5.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(!eval.evaluate_touching(&a_id, &b_id, &scene));
    }

    #[test]
    fn test_containment() {
        let inner_id = RegionId("small".to_string());
        let outer_id = RegionId("big".to_string());
        let inner = SpatialRegion::Aabb(AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]));
        let outer = SpatialRegion::Aabb(AABB::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]));

        let mut scene = SceneConfiguration::new();
        scene.regions.insert(inner_id.clone(), inner);
        scene.regions.insert(outer_id.clone(), outer);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_containment(&inner_id, &outer_id, &scene));
    }

    #[test]
    fn test_intersection() {
        let ra_id = RegionId("a".to_string());
        let rb_id = RegionId("b".to_string());
        let ra = SpatialRegion::Aabb(AABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]));
        let rb = SpatialRegion::Aabb(AABB::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]));

        let mut scene = SceneConfiguration::new();
        scene.regions.insert(ra_id.clone(), ra);
        scene.regions.insert(rb_id.clone(), rb);

        let eval = SpatialPredicateEvaluatorImpl::new();
        assert!(eval.evaluate_intersection(&ra_id, &rb_id, &scene));
    }

    #[test]
    fn test_batch_evaluate() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 3.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let predicates = vec![
            (
                SpatialPredicateId("p1".to_string()),
                SpatialPredicate::Proximity {
                    entity_a: a_id.clone(),
                    entity_b: b_id.clone(),
                    threshold: 5.0,
                },
            ),
            (
                SpatialPredicateId("p2".to_string()),
                SpatialPredicate::Proximity {
                    entity_a: a_id.clone(),
                    entity_b: b_id.clone(),
                    threshold: 0.5,
                },
            ),
        ];

        let eval = SpatialPredicateEvaluatorImpl::new();
        let valuation = eval.batch_evaluate(&predicates, &scene);

        assert_eq!(valuation.get(&SpatialPredicateId("p1".to_string())), Some(true));
        assert_eq!(valuation.get(&SpatialPredicateId("p2".to_string())), Some(false));
    }

    #[test]
    fn test_incremental_evaluate() {
        let (a_id, a_ent) = make_entity("a", 0.0, 0.0, 0.0);
        let (b_id, b_ent) = make_entity("b", 1.0, 0.0, 0.0);
        let scene = make_scene(vec![(a_id.clone(), a_ent), (b_id.clone(), b_ent)]);

        let predicates = vec![(
            SpatialPredicateId("p1".to_string()),
            SpatialPredicate::Proximity {
                entity_a: a_id.clone(),
                entity_b: b_id.clone(),
                threshold: 2.0,
            },
        )];

        let mut eval = SpatialPredicateEvaluatorImpl::new();
        let v1 = eval.incremental_evaluate(&predicates, &scene, &HashSet::new(), &HashSet::new());
        assert_eq!(v1.get(&SpatialPredicateId("p1".to_string())), Some(true));

        // No changes → should use cache.
        let v2 = eval.incremental_evaluate(&predicates, &scene, &HashSet::new(), &HashSet::new());
        assert_eq!(v2.get(&SpatialPredicateId("p1".to_string())), Some(true));
    }

    #[test]
    fn test_monotonicity_checker() {
        let checker = GeometricConsistencyChecker::new();
        let predicates = vec![
            (
                SpatialPredicateId("close".to_string()),
                SpatialPredicate::Proximity {
                    entity_a: EntityId("a".to_string()),
                    entity_b: EntityId("b".to_string()),
                    threshold: 1.0,
                },
            ),
            (
                SpatialPredicateId("far".to_string()),
                SpatialPredicate::Proximity {
                    entity_a: EntityId("a".to_string()),
                    entity_b: EntityId("b".to_string()),
                    threshold: 5.0,
                },
            ),
        ];

        // Consistent: close=true, far=true.
        let mut val = PredicateValuation::new();
        val.set(SpatialPredicateId("close".to_string()), true);
        val.set(SpatialPredicateId("far".to_string()), true);
        assert!(checker.check_monotonicity(&predicates, &val).is_empty());

        // Inconsistent: close=true, far=false.
        val.set(SpatialPredicateId("far".to_string()), false);
        let violations = checker.check_monotonicity(&predicates, &val);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_gaze_cone_sphere() {
        let eval = SpatialPredicateEvaluatorImpl::new();
        let source = Point3::new(0.0, 0.0, 0.0);
        let gaze = Vector3::new(0.0, 0.0, -1.0);
        let sphere = Sphere::new([0.0, 0.0, -5.0], 1.0);

        // Target is directly ahead.
        assert!(eval.evaluate_gaze_cone_sphere(&source, &gaze, std::f64::consts::PI / 4.0, &sphere));

        // Target is behind.
        let sphere_behind = Sphere::new([0.0, 0.0, 5.0], 1.0);
        assert!(!eval.evaluate_gaze_cone_sphere(&source, &gaze, std::f64::consts::PI / 4.0, &sphere_behind));
    }
}
