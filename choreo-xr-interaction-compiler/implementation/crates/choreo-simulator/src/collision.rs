//! CPU-based collision detection with broad-phase sweep-and-prune and
//! narrow-phase primitive tests (sphere–sphere, sphere–AABB, AABB–AABB).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::trajectory::{vec3_distance, vec3_dot, vec3_length, vec3_normalize, vec3_sub};

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollisionShape {
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Aabb {
        min: [f64; 3],
        max: [f64; 3],
    },
    Capsule {
        start: [f64; 3],
        end: [f64; 3],
        radius: f64,
    },
}

impl CollisionShape {
    /// Compute a conservative AABB enclosing the shape.
    pub fn bounding_aabb(&self) -> ([f64; 3], [f64; 3]) {
        match self {
            CollisionShape::Sphere { center, radius } => {
                let r = *radius;
                (
                    [center[0] - r, center[1] - r, center[2] - r],
                    [center[0] + r, center[1] + r, center[2] + r],
                )
            }
            CollisionShape::Aabb { min, max } => (*min, *max),
            CollisionShape::Capsule { start, end, radius } => {
                let r = *radius;
                let mn = [
                    start[0].min(end[0]) - r,
                    start[1].min(end[1]) - r,
                    start[2].min(end[2]) - r,
                ];
                let mx = [
                    start[0].max(end[0]) + r,
                    start[1].max(end[1]) + r,
                    start[2].max(end[2]) + r,
                ];
                (mn, mx)
            }
        }
    }

    pub fn center(&self) -> [f64; 3] {
        match self {
            CollisionShape::Sphere { center, .. } => *center,
            CollisionShape::Aabb { min, max } => [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ],
            CollisionShape::Capsule { start, end, .. } => [
                (start[0] + end[0]) * 0.5,
                (start[1] + end[1]) * 0.5,
                (start[2] + end[2]) * 0.5,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Collision body
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionBody {
    pub id: String,
    pub shape: CollisionShape,
    pub layer: u32,
}

// ---------------------------------------------------------------------------
// Contact results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactPoint {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub depth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactManifold {
    pub entity_a: String,
    pub entity_b: String,
    pub contacts: Vec<ContactPoint>,
    pub normal: [f64; 3],
    pub depth: f64,
}

// ---------------------------------------------------------------------------
// Collision pair (canonical ordering)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollisionPair(pub String, pub String);

impl CollisionPair {
    pub fn new(a: &str, b: &str) -> Self {
        if a <= b {
            Self(a.to_string(), b.to_string())
        } else {
            Self(b.to_string(), a.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Collision filter
// ---------------------------------------------------------------------------

/// Allows excluding certain entity pairs or layer combinations from checks.
#[derive(Debug, Clone, Default)]
pub struct CollisionFilter {
    excluded_pairs: HashSet<CollisionPair>,
    excluded_layers: HashSet<(u32, u32)>,
}

impl CollisionFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn exclude_pair(&mut self, a: &str, b: &str) {
        self.excluded_pairs.insert(CollisionPair::new(a, b));
    }

    pub fn exclude_layer_pair(&mut self, a: u32, b: u32) {
        let pair = if a <= b { (a, b) } else { (b, a) };
        self.excluded_layers.insert(pair);
    }

    pub fn is_excluded(&self, a_id: &str, a_layer: u32, b_id: &str, b_layer: u32) -> bool {
        if self.excluded_pairs.contains(&CollisionPair::new(a_id, b_id)) {
            return true;
        }
        let lp = if a_layer <= b_layer {
            (a_layer, b_layer)
        } else {
            (b_layer, a_layer)
        };
        self.excluded_layers.contains(&lp)
    }
}

// ---------------------------------------------------------------------------
// CollisionWorld
// ---------------------------------------------------------------------------

pub struct CollisionWorld {
    bodies: Vec<CollisionBody>,
    filter: CollisionFilter,
    pair_cache: HashMap<CollisionPair, bool>,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            filter: CollisionFilter::new(),
            pair_cache: HashMap::new(),
        }
    }

    pub fn with_filter(filter: CollisionFilter) -> Self {
        Self {
            bodies: Vec::new(),
            filter,
            pair_cache: HashMap::new(),
        }
    }

    pub fn set_filter(&mut self, filter: CollisionFilter) {
        self.filter = filter;
    }

    pub fn clear(&mut self) {
        self.bodies.clear();
        self.pair_cache.clear();
    }

    pub fn add_body(&mut self, body: CollisionBody) {
        self.bodies.push(body);
    }

    pub fn set_bodies(&mut self, bodies: Vec<CollisionBody>) {
        self.bodies = bodies;
        self.pair_cache.clear();
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Run full collision detection and return all contact manifolds.
    pub fn detect_collisions(&mut self) -> Vec<ContactManifold> {
        if self.bodies.len() < 2 {
            return Vec::new();
        }
        let broad = self.broad_phase();
        let mut manifolds = Vec::new();
        for (i, j) in &broad {
            let a = &self.bodies[*i];
            let b = &self.bodies[*j];
            if self.filter.is_excluded(&a.id, a.layer, &b.id, b.layer) {
                continue;
            }
            if let Some(m) = narrow_phase(&a.shape, &b.shape, &a.id, &b.id) {
                let pair = CollisionPair::new(&a.id, &b.id);
                self.pair_cache.insert(pair, true);
                manifolds.push(m);
            }
        }
        manifolds
    }

    // ---- Broad phase: sweep and prune along X axis -------------------------

    fn broad_phase(&self) -> Vec<(usize, usize)> {
        let n = self.bodies.len();
        if n < 2 {
            return Vec::new();
        }

        // Build (body_index, aabb_min_x, aabb_max_x) and sort by min_x.
        let mut entries: Vec<(usize, f64, f64, [f64; 3], [f64; 3])> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (mn, mx) = b.shape.bounding_aabb();
                (i, mn[0], mx[0], mn, mx)
            })
            .collect();

        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut pairs = Vec::new();

        for idx in 0..entries.len() {
            let (i, _, max_x_i, min_i, max_i) = entries[idx];
            for jdx in (idx + 1)..entries.len() {
                let (j, min_x_j, _, min_j, max_j) = entries[jdx];
                // If the next body's min_x exceeds our max_x, no further overlaps.
                if min_x_j > max_x_i {
                    break;
                }
                // Check Y and Z overlap.
                if aabb_overlap_axis(min_i[1], max_i[1], min_j[1], max_j[1])
                    && aabb_overlap_axis(min_i[2], max_i[2], min_j[2], max_j[2])
                {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }
}

impl Default for CollisionWorld {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Narrow-phase dispatch
// ---------------------------------------------------------------------------

fn narrow_phase(a: &CollisionShape, b: &CollisionShape, id_a: &str, id_b: &str) -> Option<ContactManifold> {
    match (a, b) {
        (CollisionShape::Sphere { .. }, CollisionShape::Sphere { .. }) => {
            sphere_sphere(a, b, id_a, id_b)
        }
        (CollisionShape::Aabb { .. }, CollisionShape::Aabb { .. }) => {
            aabb_aabb(a, b, id_a, id_b)
        }
        (CollisionShape::Sphere { .. }, CollisionShape::Aabb { .. }) => {
            sphere_aabb(a, b, id_a, id_b)
        }
        (CollisionShape::Aabb { .. }, CollisionShape::Sphere { .. }) => {
            sphere_aabb(b, a, id_b, id_a).map(|mut m| {
                m.normal = [-m.normal[0], -m.normal[1], -m.normal[2]];
                std::mem::swap(&mut m.entity_a, &mut m.entity_b);
                m
            })
        }
        // For capsule and mixed types, fall back to GJK-style delegation.
        _ => gjk_narrow_phase(a, b, id_a, id_b),
    }
}

// ---------------------------------------------------------------------------
// Sphere–Sphere
// ---------------------------------------------------------------------------

fn sphere_sphere(
    a: &CollisionShape,
    b: &CollisionShape,
    id_a: &str,
    id_b: &str,
) -> Option<ContactManifold> {
    if let (
        CollisionShape::Sphere {
            center: ca,
            radius: ra,
        },
        CollisionShape::Sphere {
            center: cb,
            radius: rb,
        },
    ) = (a, b)
    {
        let dist = vec3_distance(ca, cb);
        let sum_r = ra + rb;
        if dist >= sum_r {
            return None;
        }
        let depth = sum_r - dist;
        let normal = if dist > 1e-12 {
            vec3_normalize(&vec3_sub(cb, ca))
        } else {
            [0.0, 1.0, 0.0]
        };
        let contact_pos = [
            ca[0] + normal[0] * (ra - depth * 0.5),
            ca[1] + normal[1] * (ra - depth * 0.5),
            ca[2] + normal[2] * (ra - depth * 0.5),
        ];
        Some(ContactManifold {
            entity_a: id_a.to_string(),
            entity_b: id_b.to_string(),
            contacts: vec![ContactPoint {
                position: contact_pos,
                normal,
                depth,
            }],
            normal,
            depth,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// AABB–AABB
// ---------------------------------------------------------------------------

fn aabb_aabb(
    a: &CollisionShape,
    b: &CollisionShape,
    id_a: &str,
    id_b: &str,
) -> Option<ContactManifold> {
    if let (
        CollisionShape::Aabb {
            min: min_a,
            max: max_a,
        },
        CollisionShape::Aabb {
            min: min_b,
            max: max_b,
        },
    ) = (a, b)
    {
        // Check overlap on all three axes.
        for ax in 0..3 {
            if max_a[ax] < min_b[ax] || max_b[ax] < min_a[ax] {
                return None;
            }
        }
        // Find axis of minimum penetration.
        let mut min_depth = f64::MAX;
        let mut best_axis = 0usize;
        let mut best_sign = 1.0f64;

        for ax in 0..3 {
            let d1 = max_a[ax] - min_b[ax];
            let d2 = max_b[ax] - min_a[ax];
            if d1 < d2 {
                if d1 < min_depth {
                    min_depth = d1;
                    best_axis = ax;
                    best_sign = -1.0;
                }
            } else if d2 < min_depth {
                min_depth = d2;
                best_axis = ax;
                best_sign = 1.0;
            }
        }

        let mut normal = [0.0; 3];
        normal[best_axis] = best_sign;

        // Contact point: midpoint of overlap region.
        let mut contact = [0.0f64; 3];
        for ax in 0..3 {
            let lo = min_a[ax].max(min_b[ax]);
            let hi = max_a[ax].min(max_b[ax]);
            contact[ax] = (lo + hi) * 0.5;
        }

        Some(ContactManifold {
            entity_a: id_a.to_string(),
            entity_b: id_b.to_string(),
            contacts: vec![ContactPoint {
                position: contact,
                normal,
                depth: min_depth,
            }],
            normal,
            depth: min_depth,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Sphere–AABB
// ---------------------------------------------------------------------------

fn sphere_aabb(
    sphere: &CollisionShape,
    aabb: &CollisionShape,
    id_sphere: &str,
    id_aabb: &str,
) -> Option<ContactManifold> {
    if let (
        CollisionShape::Sphere { center, radius },
        CollisionShape::Aabb { min, max },
    ) = (sphere, aabb)
    {
        // Closest point on AABB to sphere centre.
        let closest = [
            center[0].clamp(min[0], max[0]),
            center[1].clamp(min[1], max[1]),
            center[2].clamp(min[2], max[2]),
        ];
        let diff = vec3_sub(center, &closest);
        let dist_sq = vec3_dot(&diff, &diff);
        if dist_sq >= radius * radius {
            return None;
        }
        let dist = dist_sq.sqrt();
        let depth = radius - dist;
        let normal = if dist > 1e-12 {
            vec3_normalize(&diff)
        } else {
            [0.0, 1.0, 0.0]
        };

        Some(ContactManifold {
            entity_a: id_sphere.to_string(),
            entity_b: id_aabb.to_string(),
            contacts: vec![ContactPoint {
                position: closest,
                normal,
                depth,
            }],
            normal,
            depth,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// GJK / EPA fallback for complex shapes
// ---------------------------------------------------------------------------

/// Simplified GJK-like overlap test for capsules and other unsupported pairs.
///
/// We approximate capsules as their bounding sphere and fall back to
/// sphere–sphere or sphere–AABB tests.  A full GJK implementation would
/// delegate to `choreo_spatial::gjk` at this point.
fn gjk_narrow_phase(
    a: &CollisionShape,
    b: &CollisionShape,
    id_a: &str,
    id_b: &str,
) -> Option<ContactManifold> {
    let sa = shape_to_bounding_sphere(a);
    let sb = shape_to_bounding_sphere(b);
    sphere_sphere(&sa, &sb, id_a, id_b)
}

fn shape_to_bounding_sphere(s: &CollisionShape) -> CollisionShape {
    match s {
        CollisionShape::Sphere { .. } => s.clone(),
        CollisionShape::Aabb { min, max } => {
            let center = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            let half = vec3_sub(max, min);
            let radius = vec3_length(&half) * 0.5;
            CollisionShape::Sphere { center, radius }
        }
        CollisionShape::Capsule { start, end, radius } => {
            let center = [
                (start[0] + end[0]) * 0.5,
                (start[1] + end[1]) * 0.5,
                (start[2] + end[2]) * 0.5,
            ];
            let half_len = vec3_distance(start, end) * 0.5;
            CollisionShape::Sphere {
                center,
                radius: half_len + radius,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Continuous collision detection (linear interpolation check)
// ---------------------------------------------------------------------------

/// Check whether two moving spheres will collide during `[0, 1]`.
///
/// Each sphere moves linearly from `pos_start` to `pos_end`.
/// Returns the earliest `t ∈ [0, 1]` of contact, if any.
pub fn continuous_sphere_sphere(
    c0_start: &[f64; 3],
    c0_end: &[f64; 3],
    r0: f64,
    c1_start: &[f64; 3],
    c1_end: &[f64; 3],
    r1: f64,
) -> Option<f64> {
    // Relative motion: d(t) = (c0_start + t*v0) - (c1_start + t*v1)
    //                       = rel_start + t * rel_vel
    let rel_start = vec3_sub(c0_start, c1_start);
    let v0 = vec3_sub(c0_end, c0_start);
    let v1 = vec3_sub(c1_end, c1_start);
    let rel_vel = vec3_sub(&v0, &v1);

    let sum_r = r0 + r1;

    // Solve |rel_start + t * rel_vel|^2 = sum_r^2
    let a = vec3_dot(&rel_vel, &rel_vel);
    let b = 2.0 * vec3_dot(&rel_start, &rel_vel);
    let c = vec3_dot(&rel_start, &rel_start) - sum_r * sum_r;

    if a.abs() < 1e-12 {
        // No relative motion.
        return if c <= 0.0 { Some(0.0) } else { None };
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    // Earliest root in [0, 1].
    if t1 >= 0.0 && t1 <= 1.0 {
        Some(t1)
    } else if t2 >= 0.0 && t2 <= 1.0 {
        Some(t2)
    } else if t1 < 0.0 && t2 > 0.0 {
        // Already overlapping at t=0.
        Some(0.0)
    } else {
        None
    }
}

/// Check whether two moving AABBs will overlap during `[0, 1]`.
///
/// Each AABB translates linearly by `vel` over the unit interval.
pub fn continuous_aabb_aabb(
    min_a: &[f64; 3],
    max_a: &[f64; 3],
    vel_a: &[f64; 3],
    min_b: &[f64; 3],
    max_b: &[f64; 3],
    vel_b: &[f64; 3],
) -> Option<f64> {
    let mut t_enter = 0.0f64;
    let mut t_exit = 1.0f64;

    for ax in 0..3 {
        let rel_vel = vel_a[ax] - vel_b[ax];
        let gap_low = min_b[ax] - max_a[ax]; // distance from A-max to B-min
        let gap_high = max_b[ax] - min_a[ax]; // distance from A-min to B-max

        if rel_vel.abs() < 1e-12 {
            // Static on this axis.
            if gap_low > 0.0 || gap_high < 0.0 {
                return None; // separated and won't converge
            }
        } else {
            let inv = 1.0 / rel_vel;
            let mut t0 = gap_low * inv;
            let mut t1 = gap_high * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
            if t_enter > t_exit {
                return None;
            }
        }
    }

    if t_enter <= t_exit && t_enter <= 1.0 && t_exit >= 0.0 {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn aabb_overlap_axis(min_a: f64, max_a: f64, min_b: f64, max_b: f64) -> bool {
    max_a >= min_b && max_b >= min_a
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ----- Sphere–Sphere ---------------------------------------------------

    #[test]
    fn sphere_sphere_no_collision() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Sphere {
                center: [5.0, 0.0, 0.0],
                radius: 1.0,
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    #[test]
    fn sphere_sphere_collision() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Sphere {
                center: [1.5, 0.0, 0.0],
                radius: 1.0,
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert_eq!(m.len(), 1);
        assert!(approx(m[0].depth, 0.5, 1e-9));
    }

    // ----- AABB–AABB -------------------------------------------------------

    #[test]
    fn aabb_aabb_no_collision() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Aabb {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Aabb {
                min: [2.0, 0.0, 0.0],
                max: [3.0, 1.0, 1.0],
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    #[test]
    fn aabb_aabb_overlap() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Aabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 2.0, 2.0],
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Aabb {
                min: [1.5, 0.5, 0.5],
                max: [3.0, 1.5, 1.5],
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert_eq!(m.len(), 1);
        assert!(m[0].depth > 0.0);
    }

    // ----- Sphere–AABB -----------------------------------------------------

    #[test]
    fn sphere_aabb_collision() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "s".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.5,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Aabb {
                min: [1.0, -0.5, -0.5],
                max: [2.0, 0.5, 0.5],
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn sphere_aabb_no_collision() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "s".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 0.5,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Aabb {
                min: [3.0, 0.0, 0.0],
                max: [4.0, 1.0, 1.0],
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    // ----- Collision filter ------------------------------------------------

    #[test]
    fn filter_excludes_pair() {
        let mut filter = CollisionFilter::new();
        filter.exclude_pair("a", "b");
        let mut world = CollisionWorld::with_filter(filter);
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Sphere {
                center: [1.0, 0.0, 0.0],
                radius: 2.0,
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    #[test]
    fn filter_excludes_layer() {
        let mut filter = CollisionFilter::new();
        filter.exclude_layer_pair(0, 1);
        let mut world = CollisionWorld::with_filter(filter);
        world.add_body(CollisionBody {
            id: "a".into(),
            shape: CollisionShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 2.0,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "b".into(),
            shape: CollisionShape::Sphere {
                center: [1.0, 0.0, 0.0],
                radius: 2.0,
            },
            layer: 1,
        });
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    // ----- Sweep and prune -------------------------------------------------

    #[test]
    fn broad_phase_prunes_distant() {
        let mut world = CollisionWorld::new();
        for i in 0..10 {
            world.add_body(CollisionBody {
                id: format!("s{}", i),
                shape: CollisionShape::Sphere {
                    center: [i as f64 * 100.0, 0.0, 0.0],
                    radius: 0.5,
                },
                layer: 0,
            });
        }
        let m = world.detect_collisions();
        assert!(m.is_empty());
    }

    // ----- Continuous collision ---------------------------------------------

    #[test]
    fn ccd_sphere_sphere_hit() {
        let t = continuous_sphere_sphere(
            &[0.0, 0.0, 0.0],
            &[10.0, 0.0, 0.0],
            1.0,
            &[5.0, 0.0, 0.0],
            &[5.0, 0.0, 0.0],
            1.0,
        );
        assert!(t.is_some());
        let t = t.unwrap();
        assert!(t >= 0.0 && t <= 1.0);
        // At contact, centres 2.0 apart → t where dist = 2: |10t - 5| = 2 → t = 0.3 or 0.7
        assert!(approx(t, 0.3, 0.05));
    }

    #[test]
    fn ccd_sphere_sphere_miss() {
        let t = continuous_sphere_sphere(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            0.5,
            &[5.0, 0.0, 0.0],
            &[5.0, 0.0, 0.0],
            0.5,
        );
        assert!(t.is_none());
    }

    #[test]
    fn ccd_aabb_aabb_hit() {
        let t = continuous_aabb_aabb(
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &[4.0, 0.0, 0.0],
            &[3.0, 0.0, 0.0],
            &[4.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
        );
        assert!(t.is_some());
    }

    #[test]
    fn ccd_aabb_aabb_miss() {
        let t = continuous_aabb_aabb(
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[5.0, 5.0, 5.0],
            &[6.0, 6.0, 6.0],
            &[0.0, 0.0, 0.0],
        );
        assert!(t.is_none());
    }

    // ----- Capsule (GJK fallback) ------------------------------------------

    #[test]
    fn capsule_vs_sphere_overlap() {
        let mut world = CollisionWorld::new();
        world.add_body(CollisionBody {
            id: "cap".into(),
            shape: CollisionShape::Capsule {
                start: [0.0, 0.0, 0.0],
                end: [0.0, 2.0, 0.0],
                radius: 0.5,
            },
            layer: 0,
        });
        world.add_body(CollisionBody {
            id: "sph".into(),
            shape: CollisionShape::Sphere {
                center: [0.5, 1.0, 0.0],
                radius: 0.5,
            },
            layer: 0,
        });
        let m = world.detect_collisions();
        assert_eq!(m.len(), 1);
    }

    // ----- CollisionPair ---------------------------------------------------

    #[test]
    fn collision_pair_canonical() {
        let p1 = CollisionPair::new("b", "a");
        let p2 = CollisionPair::new("a", "b");
        assert_eq!(p1, p2);
    }

    // ----- Bounding AABB ---------------------------------------------------

    #[test]
    fn bounding_aabb_sphere() {
        let s = CollisionShape::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 0.5,
        };
        let (mn, mx) = s.bounding_aabb();
        assert!(approx(mn[0], 0.5, 1e-9));
        assert!(approx(mx[0], 1.5, 1e-9));
    }
}
