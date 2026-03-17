//! GJK (Gilbert-Johnson-Keerthi) collision detection algorithm.
//!
//! Provides intersection testing, distance computation, and closest-point
//! queries between convex shapes in 3D.

use choreo_types::geometry::{AABB, OBB, Sphere, ConvexPolytope, Capsule, Point3, Vector3, Transform3D};

/// Tolerance for floating point comparisons.
const GJK_EPSILON: f64 = 1e-8;
/// Maximum number of GJK iterations.
const MAX_GJK_ITERATIONS: usize = 128;

// ─── result types ────────────────────────────────────────────────────────────

/// Result of a GJK intersection test.
#[derive(Debug, Clone)]
pub enum GjkResult {
    /// Shapes do not intersect.
    NoIntersection {
        closest_distance: f64,
        closest_points: (Point3, Point3),
    },
    /// Shapes intersect (overlap).
    Intersection {
        /// Approximate penetration depth (refined by EPA for exact value).
        penetration_depth: f64,
    },
}

impl GjkResult {
    pub fn is_intersecting(&self) -> bool {
        matches!(self, GjkResult::Intersection { .. })
    }
}

// ─── collision shapes ────────────────────────────────────────────────────────

/// A convex shape that can compute its support point in a given direction.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere(Sphere),
    Aabb(AABB),
    Obb(OBB),
    ConvexPolytope(ConvexPolytope),
    Capsule(Capsule),
}

impl CollisionShape {
    /// Compute the support point: the point on the shape farthest in `direction`.
    pub fn support(&self, direction: &Vector3) -> Point3 {
        match self {
            CollisionShape::Sphere(s) => support_sphere(s, direction),
            CollisionShape::Aabb(aabb) => support_aabb(aabb, direction),
            CollisionShape::Obb(obb) => support_obb(obb, direction),
            CollisionShape::ConvexPolytope(poly) => support_convex_polytope(poly, direction),
            CollisionShape::Capsule(cap) => support_capsule(cap, direction),
        }
    }

    /// Support point with a world transform applied.
    pub fn support_transformed(&self, transform: &Transform3D, direction: &Vector3) -> Point3 {
        let iso = transform.to_isometry();
        let inv_rot = iso.rotation.inverse();
        let local_dir = inv_rot * direction;
        let local_point = self.support(&local_dir);
        iso * local_point
    }
}

// ─── support functions ───────────────────────────────────────────────────────

fn support_sphere(sphere: &Sphere, direction: &Vector3) -> Point3 {
    let center = sphere.center_point();
    let norm = direction.norm();
    if norm < GJK_EPSILON {
        return center;
    }
    center + direction * (sphere.radius / norm)
}

fn support_aabb(aabb: &AABB, direction: &Vector3) -> Point3 {
    Point3::new(
        if direction.x >= 0.0 { aabb.max[0] } else { aabb.min[0] },
        if direction.y >= 0.0 { aabb.max[1] } else { aabb.min[1] },
        if direction.z >= 0.0 { aabb.max[2] } else { aabb.min[2] },
    )
}

fn support_obb(obb: &OBB, direction: &Vector3) -> Point3 {
    let axes = obb.axes();
    let center = obb.center_point();
    let he = obb.half_extents;
    let mut result = center;
    for i in 0..3 {
        let sign = if direction.dot(&axes[i]) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        result += axes[i] * (he[i] * sign);
    }
    result
}

fn support_convex_polytope(poly: &ConvexPolytope, direction: &Vector3) -> Point3 {
    let mut best_dot = f64::NEG_INFINITY;
    let mut best_point = Point3::origin();
    for v in &poly.vertices {
        let p = Point3::new(v[0], v[1], v[2]);
        let d = direction.dot(&p.coords);
        if d > best_dot {
            best_dot = d;
            best_point = p;
        }
    }
    best_point
}

fn support_capsule(capsule: &Capsule, direction: &Vector3) -> Point3 {
    let start = capsule.start_point();
    let end = capsule.end_point();
    let ds = direction.dot(&start.coords);
    let de = direction.dot(&end.coords);
    let base = if ds >= de { start } else { end };
    let norm = direction.norm();
    if norm < GJK_EPSILON {
        return base;
    }
    base + direction * (capsule.radius / norm)
}

// ─── simplex ─────────────────────────────────────────────────────────────────

/// A Minkowski-difference support point with witness points.
#[derive(Debug, Clone, Copy)]
pub struct SupportPoint {
    /// Point in Minkowski difference space (a - b).
    pub point: Point3,
    /// Witness on shape A.
    pub support_a: Point3,
    /// Witness on shape B.
    pub support_b: Point3,
}

/// A simplex in the GJK algorithm (1-4 vertices for 3D).
#[derive(Debug, Clone)]
pub struct Simplex {
    pub points: Vec<SupportPoint>,
}

impl Simplex {
    pub fn new() -> Self {
        Self { points: Vec::with_capacity(4) }
    }

    pub fn push(&mut self, point: SupportPoint) {
        self.points.push(point);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn last(&self) -> &SupportPoint {
        self.points.last().unwrap()
    }
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}

// ─── GJK algorithm ──────────────────────────────────────────────────────────

fn minkowski_support(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
    direction: &Vector3,
) -> SupportPoint {
    let sa = shape_a.support_transformed(transform_a, direction);
    let sb = shape_b.support_transformed(transform_b, &(-direction));
    SupportPoint {
        point: Point3::from(sa.coords - sb.coords),
        support_a: sa,
        support_b: sb,
    }
}

/// Test whether two convex shapes intersect.
pub fn gjk_intersection(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
) -> GjkResult {
    let center_a = transform_a.position_point();
    let center_b = transform_b.position_point();
    let mut direction = center_b - center_a;
    if direction.norm_squared() < GJK_EPSILON * GJK_EPSILON {
        direction = Vector3::new(1.0, 0.0, 0.0);
    }

    let mut simplex = Simplex::new();
    let first = minkowski_support(shape_a, transform_a, shape_b, transform_b, &direction);
    simplex.push(first);
    direction = -first.point.coords;

    for _ in 0..MAX_GJK_ITERATIONS {
        if direction.norm_squared() < GJK_EPSILON * GJK_EPSILON {
            return GjkResult::Intersection {
                penetration_depth: 0.0,
            };
        }

        let new_point = minkowski_support(shape_a, transform_a, shape_b, transform_b, &direction);

        if new_point.point.coords.dot(&direction) < -GJK_EPSILON {
            // The new support point didn't pass the origin → no intersection.
            let (dist, cp_a, cp_b) = closest_points_from_simplex(&simplex);
            return GjkResult::NoIntersection {
                closest_distance: dist,
                closest_points: (cp_a, cp_b),
            };
        }

        simplex.push(new_point);

        let (contains_origin, new_simplex, new_dir) = do_simplex(simplex);
        simplex = new_simplex;
        direction = new_dir;

        if contains_origin {
            return GjkResult::Intersection {
                penetration_depth: 0.0,
            };
        }
    }

    // Didn't converge → assume no intersection with the distance from current simplex.
    let (dist, cp_a, cp_b) = closest_points_from_simplex(&simplex);
    GjkResult::NoIntersection {
        closest_distance: dist,
        closest_points: (cp_a, cp_b),
    }
}

/// Compute the distance between two non-intersecting convex shapes.
pub fn gjk_distance(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
) -> f64 {
    match gjk_intersection(shape_a, transform_a, shape_b, transform_b) {
        GjkResult::NoIntersection {
            closest_distance, ..
        } => closest_distance,
        GjkResult::Intersection { .. } => 0.0,
    }
}

/// Compute closest points between two non-intersecting convex shapes.
pub fn gjk_closest_points(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
) -> (Point3, Point3) {
    match gjk_intersection(shape_a, transform_a, shape_b, transform_b) {
        GjkResult::NoIntersection {
            closest_points, ..
        } => closest_points,
        GjkResult::Intersection { .. } => {
            let ca = transform_a.position_point();
            let cb = transform_b.position_point();
            (ca, cb)
        }
    }
}

// ─── do_simplex ──────────────────────────────────────────────────────────────

/// Process the simplex: determine if origin is enclosed and update direction.
pub fn do_simplex(simplex: Simplex) -> (bool, Simplex, Vector3) {
    match simplex.len() {
        2 => do_simplex_line(simplex),
        3 => do_simplex_triangle(simplex),
        4 => do_simplex_tetrahedron(simplex),
        _ => (false, simplex, Vector3::zeros()),
    }
}

fn do_simplex_line(simplex: Simplex) -> (bool, Simplex, Vector3) {
    let a = simplex.points[1]; // most recently added
    let b = simplex.points[0];
    let ab = b.point - a.point;
    let ao = Point3::origin() - a.point;

    if ab.dot(&ao) > 0.0 {
        // Origin is in the region between A and B.
        let dir = triple_cross(&ab, &ao, &ab);
        (false, simplex, dir)
    } else {
        // Origin is beyond A.
        let mut new_simplex = Simplex::new();
        new_simplex.push(a);
        (false, new_simplex, ao)
    }
}

fn do_simplex_triangle(simplex: Simplex) -> (bool, Simplex, Vector3) {
    let a = simplex.points[2]; // most recently added
    let b = simplex.points[1];
    let c = simplex.points[0];

    let ab = b.point - a.point;
    let ac = c.point - a.point;
    let ao = Point3::origin() - a.point;
    let abc = ab.cross(&ac);

    // Check which side of triangle edges the origin lies.
    let abc_cross_ac = abc.cross(&ac);
    if abc_cross_ac.dot(&ao) > 0.0 {
        if ac.dot(&ao) > 0.0 {
            // Region AC.
            let mut s = Simplex::new();
            s.push(c);
            s.push(a);
            let dir = triple_cross(&ac, &ao, &ac);
            return (false, s, dir);
        } else {
            return do_simplex_line_ab(a, b, ao);
        }
    }

    let ab_cross_abc = ab.cross(&abc);
    if ab_cross_abc.dot(&ao) > 0.0 {
        return do_simplex_line_ab(a, b, ao);
    }

    // Origin is within the triangle.
    if abc.dot(&ao) > 0.0 {
        // Above triangle.
        (false, simplex, abc)
    } else {
        // Below triangle: reverse winding.
        let mut s = Simplex::new();
        s.push(b);
        s.push(c);
        s.push(a);
        (false, s, -abc)
    }
}

fn do_simplex_line_ab(a: SupportPoint, b: SupportPoint, ao: Vector3) -> (bool, Simplex, Vector3) {
    let ab = b.point - a.point;
    if ab.dot(&ao) > 0.0 {
        let mut s = Simplex::new();
        s.push(b);
        s.push(a);
        let dir = triple_cross(&ab, &ao, &ab);
        (false, s, dir)
    } else {
        let mut s = Simplex::new();
        s.push(a);
        (false, s, ao)
    }
}

fn do_simplex_tetrahedron(simplex: Simplex) -> (bool, Simplex, Vector3) {
    let a = simplex.points[3]; // most recently added
    let b = simplex.points[2];
    let c = simplex.points[1];
    let d = simplex.points[0];

    let ab = b.point - a.point;
    let ac = c.point - a.point;
    let ad = d.point - a.point;
    let ao = Point3::origin() - a.point;

    let abc = ab.cross(&ac);
    let acd = ac.cross(&ad);
    let adb = ad.cross(&ab);

    // Check each face of the tetrahedron.
    let abc_test = abc.dot(&ao) > 0.0;
    let acd_test = acd.dot(&ao) > 0.0;
    let adb_test = adb.dot(&ao) > 0.0;

    if abc_test {
        // Origin is on the ABC side.
        let mut s = Simplex::new();
        s.push(c);
        s.push(b);
        s.push(a);
        return do_simplex_triangle(s);
    }

    if acd_test {
        let mut s = Simplex::new();
        s.push(d);
        s.push(c);
        s.push(a);
        return do_simplex_triangle(s);
    }

    if adb_test {
        let mut s = Simplex::new();
        s.push(b);
        s.push(d);
        s.push(a);
        return do_simplex_triangle(s);
    }

    // Origin is inside the tetrahedron.
    (true, simplex, Vector3::zeros())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Triple cross product: (a × b) × c
fn triple_cross(a: &Vector3, b: &Vector3, c: &Vector3) -> Vector3 {
    let ab = a.cross(b);
    ab.cross(c)
}

/// Extract closest points from current simplex.
fn closest_points_from_simplex(simplex: &Simplex) -> (f64, Point3, Point3) {
    match simplex.len() {
        0 => (f64::INFINITY, Point3::origin(), Point3::origin()),
        1 => {
            let p = &simplex.points[0];
            let dist = p.point.coords.norm();
            (dist, p.support_a, p.support_b)
        }
        2 => closest_point_on_line_segment(&simplex.points[0], &simplex.points[1]),
        3 => closest_point_on_triangle(
            &simplex.points[0],
            &simplex.points[1],
            &simplex.points[2],
        ),
        _ => {
            // For tetrahedron, find closest face.
            let mut best_dist = f64::INFINITY;
            let mut best_a = Point3::origin();
            let mut best_b = Point3::origin();
            let faces = [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ];
            for face in &faces {
                let (d, a, b) = closest_point_on_triangle(
                    &simplex.points[face[0]],
                    &simplex.points[face[1]],
                    &simplex.points[face[2]],
                );
                if d < best_dist {
                    best_dist = d;
                    best_a = a;
                    best_b = b;
                }
            }
            (best_dist, best_a, best_b)
        }
    }
}

fn closest_point_on_line_segment(
    p0: &SupportPoint,
    p1: &SupportPoint,
) -> (f64, Point3, Point3) {
    let ab = p1.point - p0.point;
    let ao = -p0.point.coords;
    let ab_sq = ab.norm_squared();
    if ab_sq < GJK_EPSILON * GJK_EPSILON {
        let dist = p0.point.coords.norm();
        return (dist, p0.support_a, p0.support_b);
    }
    let t = (ao.dot(&ab) / ab_sq).clamp(0.0, 1.0);
    let closest = p0.point + ab * t;
    let dist = closest.coords.norm();
    let witness_a = Point3::from(p0.support_a.coords * (1.0 - t) + p1.support_a.coords * t);
    let witness_b = Point3::from(p0.support_b.coords * (1.0 - t) + p1.support_b.coords * t);
    (dist, witness_a, witness_b)
}

fn closest_point_on_triangle(
    p0: &SupportPoint,
    p1: &SupportPoint,
    p2: &SupportPoint,
) -> (f64, Point3, Point3) {
    let a = p0.point;
    let b = p1.point;
    let c = p2.point;

    let ab = b - a;
    let ac = c - a;
    let ao = Point3::origin() - a;

    let d1 = ab.dot(&ao);
    let d2 = ac.dot(&ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        let dist = a.coords.norm();
        return (dist, p0.support_a, p0.support_b);
    }

    let bo = Point3::origin() - b;
    let d3 = ab.dot(&bo);
    let d4 = ac.dot(&bo);
    if d3 >= 0.0 && d4 <= d3 {
        let dist = b.coords.norm();
        return (dist, p1.support_a, p1.support_b);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + ab * v;
        let dist = closest.coords.norm();
        let wa = Point3::from(p0.support_a.coords * (1.0 - v) + p1.support_a.coords * v);
        let wb = Point3::from(p0.support_b.coords * (1.0 - v) + p1.support_b.coords * v);
        return (dist, wa, wb);
    }

    let co = Point3::origin() - c;
    let d5 = ab.dot(&co);
    let d6 = ac.dot(&co);
    if d6 >= 0.0 && d5 <= d6 {
        let dist = c.coords.norm();
        return (dist, p2.support_a, p2.support_b);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + ac * w;
        let dist = closest.coords.norm();
        let wa = Point3::from(p0.support_a.coords * (1.0 - w) + p2.support_a.coords * w);
        let wb = Point3::from(p0.support_b.coords * (1.0 - w) + p2.support_b.coords * w);
        return (dist, wa, wb);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + (c - b) * w;
        let dist = closest.coords.norm();
        let wa = Point3::from(p1.support_a.coords * (1.0 - w) + p2.support_a.coords * w);
        let wb = Point3::from(p1.support_b.coords * (1.0 - w) + p2.support_b.coords * w);
        return (dist, wa, wb);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    let dist = closest.coords.norm();
    let u = 1.0 - v - w;
    let wa = Point3::from(p0.support_a.coords * u + p1.support_a.coords * v + p2.support_a.coords * w);
    let wb = Point3::from(p0.support_b.coords * u + p1.support_b.coords * v + p2.support_b.coords * w);
    (dist, wa, wb)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Transform3D {
        Transform3D::identity()
    }

    fn at(x: f64, y: f64, z: f64) -> Transform3D {
        Transform3D::from_position(x, y, z)
    }

    fn unit_sphere() -> CollisionShape {
        CollisionShape::Sphere(Sphere::new([0.0, 0.0, 0.0], 1.0))
    }

    fn unit_aabb() -> CollisionShape {
        CollisionShape::Aabb(AABB::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]))
    }

    #[test]
    fn test_sphere_sphere_intersecting() {
        let s = unit_sphere();
        let result = gjk_intersection(&s, &at(0.0, 0.0, 0.0), &s, &at(1.0, 0.0, 0.0));
        assert!(result.is_intersecting());
    }

    #[test]
    fn test_sphere_sphere_separated() {
        let s = unit_sphere();
        let result = gjk_intersection(&s, &at(0.0, 0.0, 0.0), &s, &at(3.0, 0.0, 0.0));
        assert!(!result.is_intersecting());
        if let GjkResult::NoIntersection { closest_distance, .. } = result {
            assert!((closest_distance - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_sphere_sphere_touching() {
        let s = unit_sphere();
        let result = gjk_intersection(&s, &at(0.0, 0.0, 0.0), &s, &at(2.0, 0.0, 0.0));
        // At exactly touching, either result is acceptable.
        let dist = gjk_distance(&s, &at(0.0, 0.0, 0.0), &s, &at(2.0, 0.0, 0.0));
        assert!(dist < 0.1);
    }

    #[test]
    fn test_aabb_aabb_intersecting() {
        let a = unit_aabb();
        let result = gjk_intersection(&a, &at(0.0, 0.0, 0.0), &a, &at(0.5, 0.0, 0.0));
        assert!(result.is_intersecting());
    }

    #[test]
    fn test_aabb_aabb_separated() {
        let a = unit_aabb();
        let result = gjk_intersection(&a, &at(0.0, 0.0, 0.0), &a, &at(2.0, 0.0, 0.0));
        assert!(!result.is_intersecting());
        if let GjkResult::NoIntersection { closest_distance, .. } = result {
            assert!((closest_distance - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_sphere_aabb_intersecting() {
        let s = unit_sphere();
        let a = unit_aabb();
        let result = gjk_intersection(&s, &at(0.0, 0.0, 0.0), &a, &at(1.0, 0.0, 0.0));
        assert!(result.is_intersecting());
    }

    #[test]
    fn test_sphere_aabb_separated() {
        let s = unit_sphere();
        let a = unit_aabb();
        let result = gjk_intersection(&s, &at(0.0, 0.0, 0.0), &a, &at(3.0, 0.0, 0.0));
        assert!(!result.is_intersecting());
    }

    #[test]
    fn test_gjk_distance() {
        let s = unit_sphere();
        let d = gjk_distance(&s, &at(0.0, 0.0, 0.0), &s, &at(4.0, 0.0, 0.0));
        assert!((d - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_gjk_closest_points() {
        let s = unit_sphere();
        let (pa, pb) = gjk_closest_points(&s, &at(0.0, 0.0, 0.0), &s, &at(4.0, 0.0, 0.0));
        assert!((pa.x - 1.0).abs() < 0.2);
        assert!((pb.x - 3.0).abs() < 0.2);
    }

    #[test]
    fn test_convex_polytope_support() {
        // Unit cube as convex polytope.
        let poly = ConvexPolytope::new(
            vec![
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![],
        );
        let shape = CollisionShape::ConvexPolytope(poly);
        let p = shape.support(&Vector3::new(1.0, 1.0, 1.0));
        assert!((p.x - 1.0).abs() < GJK_EPSILON);
        assert!((p.y - 1.0).abs() < GJK_EPSILON);
        assert!((p.z - 1.0).abs() < GJK_EPSILON);
    }

    #[test]
    fn test_capsule_support() {
        let cap = Capsule {
            start: [0.0, 0.0, 0.0],
            end: [0.0, 2.0, 0.0],
            radius: 0.5,
        };
        let shape = CollisionShape::Capsule(cap);
        let p = shape.support(&Vector3::new(0.0, 1.0, 0.0));
        assert!((p.y - 2.5).abs() < GJK_EPSILON);
    }

    #[test]
    fn test_identical_shapes() {
        let s = unit_sphere();
        let result = gjk_intersection(&s, &identity(), &s, &identity());
        assert!(result.is_intersecting());
    }

    #[test]
    fn test_obb_intersection() {
        let obb = CollisionShape::Obb(OBB {
            center: [0.0, 0.0, 0.0],
            half_extents: [1.0, 1.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
        });
        let result = gjk_intersection(&obb, &identity(), &obb, &at(1.0, 0.0, 0.0));
        assert!(result.is_intersecting());
    }
}
