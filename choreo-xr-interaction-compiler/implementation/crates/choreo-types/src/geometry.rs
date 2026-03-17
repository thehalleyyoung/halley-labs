//! Geometric primitives and spatial data types.

use serde::{Deserialize, Serialize};
use nalgebra as na;

/// 3D point.
pub type Point3 = na::Point3<f64>;

/// 3D vector.
pub type Vector3 = na::Vector3<f64>;

/// Unit quaternion for rotations.
pub type Quaternion = na::UnitQuaternion<f64>;

/// 3D rigid-body transform: rotation + translation + uniform scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform3D {
    pub position: [f64; 3],
    pub rotation: [f64; 4], // (x, y, z, w) quaternion
    pub scale: [f64; 3],
}

impl Default for Transform3D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform3D {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn from_position(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
            ..Default::default()
        }
    }

    pub fn position_vec(&self) -> Vector3 {
        Vector3::new(self.position[0], self.position[1], self.position[2])
    }

    pub fn position_point(&self) -> Point3 {
        Point3::new(self.position[0], self.position[1], self.position[2])
    }

    pub fn quaternion(&self) -> Quaternion {
        let q = na::Quaternion::new(
            self.rotation[3],
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
        );
        na::UnitQuaternion::from_quaternion(q)
    }

    pub fn rotation_matrix(&self) -> na::Matrix3<f64> {
        *self.quaternion().to_rotation_matrix().matrix()
    }

    pub fn to_isometry(&self) -> na::Isometry3<f64> {
        na::Isometry3::from_parts(
            na::Translation3::new(self.position[0], self.position[1], self.position[2]),
            self.quaternion(),
        )
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AABB {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl AABB {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Point3]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in points {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        Self { min, max }
    }

    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        )
    }

    pub fn half_extents(&self) -> Vector3 {
        Vector3::new(
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        )
    }

    pub fn extents(&self) -> Vector3 {
        Vector3::new(
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )
    }

    pub fn volume(&self) -> f64 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    pub fn surface_area(&self) -> f64 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    pub fn contains_point(&self, p: &Point3) -> bool {
        p.x >= self.min[0]
            && p.x <= self.max[0]
            && p.y >= self.min[1]
            && p.y <= self.max[1]
            && p.z >= self.min[2]
            && p.z <= self.max[2]
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    pub fn contains_aabb(&self, other: &AABB) -> bool {
        self.min[0] <= other.min[0]
            && self.max[0] >= other.max[0]
            && self.min[1] <= other.min[1]
            && self.max[1] >= other.max[1]
            && self.min[2] <= other.min[2]
            && self.max[2] >= other.max[2]
    }

    pub fn merge(&self, other: &AABB) -> AABB {
        AABB {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn expand(&self, amount: f64) -> AABB {
        AABB {
            min: [
                self.min[0] - amount,
                self.min[1] - amount,
                self.min[2] - amount,
            ],
            max: [
                self.max[0] + amount,
                self.max[1] + amount,
                self.max[2] + amount,
            ],
        }
    }

    pub fn empty() -> Self {
        AABB {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }

    pub fn min_point(&self) -> Point3 {
        Point3::new(self.min[0], self.min[1], self.min[2])
    }

    pub fn max_point(&self) -> Point3 {
        Point3::new(self.max[0], self.max[1], self.max[2])
    }

    /// Get the 8 corner vertices of this AABB.
    pub fn corners(&self) -> [Point3; 8] {
        [
            Point3::new(self.min[0], self.min[1], self.min[2]),
            Point3::new(self.max[0], self.min[1], self.min[2]),
            Point3::new(self.min[0], self.max[1], self.min[2]),
            Point3::new(self.max[0], self.max[1], self.min[2]),
            Point3::new(self.min[0], self.min[1], self.max[2]),
            Point3::new(self.max[0], self.min[1], self.max[2]),
            Point3::new(self.min[0], self.max[1], self.max[2]),
            Point3::new(self.max[0], self.max[1], self.max[2]),
        ]
    }

    /// Distance from a point to the closest point on the AABB surface.
    pub fn distance_to_point(&self, p: &Point3) -> f64 {
        let mut dist_sq = 0.0;
        for i in 0..3 {
            let v = p[i];
            if v < self.min[i] {
                dist_sq += (self.min[i] - v) * (self.min[i] - v);
            } else if v > self.max[i] {
                dist_sq += (v - self.max[i]) * (v - self.max[i]);
            }
        }
        dist_sq.sqrt()
    }

    /// Intersect ray with AABB, returning (tmin, tmax) or None.
    pub fn ray_intersect(&self, origin: &Point3, dir: &Vector3) -> Option<(f64, f64)> {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;
        for i in 0..3 {
            if dir[i].abs() < 1e-12 {
                if origin[i] < self.min[i] || origin[i] > self.max[i] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir[i];
                let mut t1 = (self.min[i] - origin[i]) * inv_d;
                let mut t2 = (self.max[i] - origin[i]) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                tmin = tmin.max(t1);
                tmax = tmax.min(t2);
                if tmin > tmax {
                    return None;
                }
            }
        }
        if tmax < 0.0 {
            return None;
        }
        Some((tmin.max(0.0), tmax))
    }
}

/// Oriented bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OBB {
    pub center: [f64; 3],
    pub half_extents: [f64; 3],
    pub orientation: [f64; 4], // quaternion (x, y, z, w)
}

impl OBB {
    pub fn center_point(&self) -> Point3 {
        Point3::new(self.center[0], self.center[1], self.center[2])
    }

    pub fn half_extents_vec(&self) -> Vector3 {
        Vector3::new(self.half_extents[0], self.half_extents[1], self.half_extents[2])
    }

    pub fn axes(&self) -> [Vector3; 3] {
        let q = na::Quaternion::new(
            self.orientation[3],
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
        );
        let uq = na::UnitQuaternion::from_quaternion(q);
        let mat = *uq.to_rotation_matrix().matrix();
        [
            Vector3::new(mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]),
            Vector3::new(mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]),
            Vector3::new(mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]),
        ]
    }

    pub fn to_aabb(&self) -> AABB {
        let axes = self.axes();
        let mut half = Vector3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                half[i] += (axes[j][i] * self.half_extents[j]).abs();
            }
        }
        let c = self.center_point();
        AABB::new(
            [(c.x - half.x), (c.y - half.y), (c.z - half.z)],
            [(c.x + half.x), (c.y + half.y), (c.z + half.z)],
        )
    }

    pub fn corners(&self) -> [Point3; 8] {
        let axes = self.axes();
        let c = self.center_point();
        let he = self.half_extents;
        let mut corners = [Point3::origin(); 8];
        let signs: [(f64, f64, f64); 8] = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ];
        for (idx, &(sx, sy, sz)) in signs.iter().enumerate() {
            corners[idx] = c + axes[0] * (he[0] * sx) + axes[1] * (he[1] * sy) + axes[2] * (he[2] * sz);
        }
        corners
    }
}

/// Sphere.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Sphere {
    pub center: [f64; 3],
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn center_point(&self) -> Point3 {
        Point3::new(self.center[0], self.center[1], self.center[2])
    }

    pub fn to_aabb(&self) -> AABB {
        AABB::new(
            [
                self.center[0] - self.radius,
                self.center[1] - self.radius,
                self.center[2] - self.radius,
            ],
            [
                self.center[0] + self.radius,
                self.center[1] + self.radius,
                self.center[2] + self.radius,
            ],
        )
    }

    pub fn contains_point(&self, p: &Point3) -> bool {
        let dx = p.x - self.center[0];
        let dy = p.y - self.center[1];
        let dz = p.z - self.center[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }
}

/// Capsule (swept sphere along a line segment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capsule {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub radius: f64,
}

impl Capsule {
    pub fn start_point(&self) -> Point3 {
        Point3::new(self.start[0], self.start[1], self.start[2])
    }

    pub fn end_point(&self) -> Point3 {
        Point3::new(self.end[0], self.end[1], self.end[2])
    }

    pub fn to_aabb(&self) -> AABB {
        AABB::new(
            [
                self.start[0].min(self.end[0]) - self.radius,
                self.start[1].min(self.end[1]) - self.radius,
                self.start[2].min(self.end[2]) - self.radius,
            ],
            [
                self.start[0].max(self.end[0]) + self.radius,
                self.start[1].max(self.end[1]) + self.radius,
                self.start[2].max(self.end[2]) + self.radius,
            ],
        )
    }
}

/// Convex hull represented by a set of points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvexHull {
    pub points: Vec<[f64; 3]>,
}

impl ConvexHull {
    pub fn new(points: Vec<[f64; 3]>) -> Self {
        Self { points }
    }

    pub fn to_aabb(&self) -> AABB {
        AABB::from_points(
            &self
                .points
                .iter()
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect::<Vec<_>>(),
        )
    }
}

/// Convex polytope with vertices and face indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvexPolytope {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
}

impl ConvexPolytope {
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<Vec<usize>>) -> Self {
        Self { vertices, faces }
    }

    pub fn to_aabb(&self) -> AABB {
        AABB::from_points(
            &self
                .vertices
                .iter()
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect::<Vec<_>>(),
        )
    }
}

/// A 3D plane defined by normal·x = d.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Plane {
    pub normal: [f64; 3],
    pub d: f64,
}

impl Plane {
    pub fn new(normal: Vector3, d: f64) -> Self {
        Self {
            normal: [normal.x, normal.y, normal.z],
            d,
        }
    }

    pub fn normal_vec(&self) -> Vector3 {
        Vector3::new(self.normal[0], self.normal[1], self.normal[2])
    }

    pub fn signed_distance(&self, p: &Point3) -> f64 {
        let n = self.normal_vec();
        n.dot(&p.coords) - self.d
    }
}

/// A 3D ray: origin + direction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

impl Ray {
    pub fn new(origin: Point3, direction: Vector3) -> Self {
        Self {
            origin: [origin.x, origin.y, origin.z],
            direction: [direction.x, direction.y, direction.z],
        }
    }

    pub fn origin_point(&self) -> Point3 {
        Point3::new(self.origin[0], self.origin[1], self.origin[2])
    }

    pub fn direction_vec(&self) -> Vector3 {
        Vector3::new(self.direction[0], self.direction[1], self.direction[2])
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin_point() + self.direction_vec() * t
    }
}

/// A 3D line segment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LineSegment {
    pub start: [f64; 3],
    pub end: [f64; 3],
}

/// Bounding volume trait.
pub trait BoundingVolume {
    fn aabb(&self) -> AABB;
    fn contains_point(&self, point: &Point3) -> bool;
}

impl BoundingVolume for AABB {
    fn aabb(&self) -> AABB {
        *self
    }
    fn contains_point(&self, point: &Point3) -> bool {
        self.contains_point(point)
    }
}

impl BoundingVolume for Sphere {
    fn aabb(&self) -> AABB {
        self.to_aabb()
    }
    fn contains_point(&self, point: &Point3) -> bool {
        self.contains_point(point)
    }
}

/// Spatial relation between two objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialRelation {
    Disjoint,
    Touching,
    Overlapping,
    Contains,
    ContainedBy,
    Equal,
}
