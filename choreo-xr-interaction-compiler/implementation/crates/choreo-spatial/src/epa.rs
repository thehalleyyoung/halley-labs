//! EPA (Expanding Polytope Algorithm) for computing penetration depth.
//!
//! Given an initial GJK simplex that encloses the origin, EPA iteratively
//! expands a convex polytope to find the minimum translation vector (MTV).

use choreo_types::geometry::{Point3, Vector3, Transform3D};
use crate::gjk::{CollisionShape, Simplex, SupportPoint};

const EPA_EPSILON: f64 = 1e-6;
const MAX_EPA_ITERATIONS: usize = 64;
const MAX_EPA_FACES: usize = 256;

// ─── result ──────────────────────────────────────────────────────────────────

/// Result of the EPA algorithm.
#[derive(Debug, Clone)]
pub struct EpaResult {
    /// Penetration depth (minimum translation distance).
    pub penetration_depth: f64,
    /// Penetration normal (direction to separate shape A from shape B).
    pub penetration_normal: Vector3,
    /// Contact point on shape A.
    pub contact_point_a: Point3,
    /// Contact point on shape B.
    pub contact_point_b: Point3,
}

// ─── polytope face ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct EpaFace {
    /// Indices into the polytope vertex buffer.
    indices: [usize; 3],
    /// Outward-facing normal.
    normal: Vector3,
    /// Distance from origin to the face plane.
    distance: f64,
}

impl EpaFace {
    fn new(vertices: &[SupportPoint], i0: usize, i1: usize, i2: usize) -> Option<Self> {
        let a = vertices[i0].point;
        let b = vertices[i1].point;
        let c = vertices[i2].point;
        let ab = b - a;
        let ac = c - a;
        let normal = ab.cross(&ac);
        let len = normal.norm();
        if len < EPA_EPSILON {
            return None;
        }
        let normal = normal / len;
        let distance = normal.dot(&a.coords);
        // Ensure normal points away from origin.
        if distance < 0.0 {
            Some(EpaFace {
                indices: [i0, i2, i1],
                normal: -normal,
                distance: -distance,
            })
        } else {
            Some(EpaFace {
                indices: [i0, i1, i2],
                normal,
                distance,
            })
        }
    }
}

// ─── edge ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct EpaEdge {
    a: usize,
    b: usize,
}

impl EpaEdge {
    fn new(a: usize, b: usize) -> Self {
        Self { a, b }
    }

    fn reversed(&self) -> Self {
        Self {
            a: self.b,
            b: self.a,
        }
    }
}

// ─── polytope ────────────────────────────────────────────────────────────────

/// The expanding polytope used by EPA.
#[derive(Debug)]
pub struct EpaPolytope {
    pub vertices: Vec<SupportPoint>,
    faces: Vec<EpaFace>,
}

impl EpaPolytope {
    /// Build from an initial GJK tetrahedron simplex.
    fn from_simplex(simplex: &Simplex) -> Option<Self> {
        if simplex.len() < 4 {
            return None;
        }
        let vertices = simplex.points.clone();
        let face_indices = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ];
        let mut faces = Vec::new();
        for fi in &face_indices {
            if let Some(face) = EpaFace::new(&vertices, fi[0], fi[1], fi[2]) {
                faces.push(face);
            }
        }
        if faces.len() < 4 {
            return None;
        }
        Some(Self { vertices, faces })
    }

    /// Find the face closest to the origin.
    fn closest_face(&self) -> (usize, f64, Vector3) {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        let mut best_normal = Vector3::zeros();
        for (i, face) in self.faces.iter().enumerate() {
            if face.distance < best_dist {
                best_dist = face.distance;
                best_normal = face.normal;
                best_idx = i;
            }
        }
        (best_idx, best_dist, best_normal)
    }

    /// Expand the polytope by adding a new support point.
    fn expand(&mut self, new_point: SupportPoint) -> bool {
        let new_idx = self.vertices.len();
        self.vertices.push(new_point);

        // Find all faces visible from the new point.
        let mut visible = Vec::new();
        for (i, face) in self.faces.iter().enumerate() {
            let v = self.vertices[new_idx].point.coords - self.vertices[face.indices[0]].point.coords;
            if face.normal.dot(&v) > EPA_EPSILON {
                visible.push(i);
            }
        }

        if visible.is_empty() {
            self.vertices.pop();
            return false;
        }

        // Collect horizon edges (edges shared by exactly one visible face).
        let mut edges: Vec<EpaEdge> = Vec::new();
        for &fi in &visible {
            let face = &self.faces[fi];
            let face_edges = [
                EpaEdge::new(face.indices[0], face.indices[1]),
                EpaEdge::new(face.indices[1], face.indices[2]),
                EpaEdge::new(face.indices[2], face.indices[0]),
            ];
            for edge in &face_edges {
                let rev = edge.reversed();
                if let Some(pos) = edges.iter().position(|e| *e == rev) {
                    edges.remove(pos);
                } else {
                    edges.push(edge.clone());
                }
            }
        }

        // Remove visible faces (in reverse order to preserve indices).
        let mut visible_sorted = visible;
        visible_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for idx in visible_sorted {
            self.faces.swap_remove(idx);
        }

        // Create new faces from horizon edges to new point.
        for edge in &edges {
            if let Some(face) = EpaFace::new(&self.vertices, edge.a, edge.b, new_idx) {
                if self.faces.len() < MAX_EPA_FACES {
                    self.faces.push(face);
                }
            }
        }

        true
    }
}

// ─── EPA algorithm ───────────────────────────────────────────────────────────

/// Compute penetration depth and contact information using EPA.
///
/// `initial_simplex` must be the simplex from GJK that encloses the origin
/// (i.e., GJK returned `Intersection`). The simplex must have 4 points.
pub fn epa_penetration(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
    initial_simplex: &Simplex,
) -> Option<EpaResult> {
    // Ensure we have a tetrahedron. If not, try to build one.
    let simplex = ensure_tetrahedron(shape_a, transform_a, shape_b, transform_b, initial_simplex)?;
    let mut polytope = EpaPolytope::from_simplex(&simplex)?;

    for _ in 0..MAX_EPA_ITERATIONS {
        let (closest_idx, closest_dist, closest_normal) = polytope.closest_face();

        // Get new support point in the direction of the closest face normal.
        let new_support = minkowski_support(shape_a, transform_a, shape_b, transform_b, &closest_normal);

        let new_dist = new_support.point.coords.dot(&closest_normal);

        // Convergence check.
        if (new_dist - closest_dist).abs() < EPA_EPSILON {
            let (contact_a, contact_b) = compute_contact_points(&polytope, closest_idx);
            return Some(EpaResult {
                penetration_depth: closest_dist,
                penetration_normal: closest_normal,
                contact_point_a: contact_a,
                contact_point_b: contact_b,
            });
        }

        if !polytope.expand(new_support) {
            let (contact_a, contact_b) = compute_contact_points(&polytope, closest_idx);
            return Some(EpaResult {
                penetration_depth: closest_dist,
                penetration_normal: closest_normal,
                contact_point_a: contact_a,
                contact_point_b: contact_b,
            });
        }
    }

    // Return best estimate.
    let (closest_idx, closest_dist, closest_normal) = polytope.closest_face();
    let (contact_a, contact_b) = compute_contact_points(&polytope, closest_idx);
    Some(EpaResult {
        penetration_depth: closest_dist,
        penetration_normal: closest_normal,
        contact_point_a: contact_a,
        contact_point_b: contact_b,
    })
}

/// Ensure the simplex has 4 points (tetrahedron) for EPA.
fn ensure_tetrahedron(
    shape_a: &CollisionShape,
    transform_a: &Transform3D,
    shape_b: &CollisionShape,
    transform_b: &Transform3D,
    simplex: &Simplex,
) -> Option<Simplex> {
    if simplex.len() >= 4 {
        return Some(simplex.clone());
    }

    let mut result = simplex.clone();

    // Add points in orthogonal directions until we have 4.
    let search_dirs = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 1.0),
        Vector3::new(1.0, 0.0, 1.0),
        Vector3::new(-1.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 1.0),
        Vector3::new(1.0, 0.0, -1.0),
    ];

    for dir in &search_dirs {
        if result.len() >= 4 {
            break;
        }
        let sp = minkowski_support(shape_a, transform_a, shape_b, transform_b, dir);
        // Check it's not duplicate.
        let is_dup = result
            .points
            .iter()
            .any(|p| (p.point - sp.point).norm() < EPA_EPSILON);
        if !is_dup {
            result.push(sp);
        }
    }

    if result.len() >= 4 {
        Some(result)
    } else {
        None
    }
}

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

/// Compute contact points from the closest face using barycentric coordinates.
fn compute_contact_points(polytope: &EpaPolytope, face_idx: usize) -> (Point3, Point3) {
    if face_idx >= polytope.faces.len() {
        return (Point3::origin(), Point3::origin());
    }
    let face = &polytope.faces[face_idx];
    let a = polytope.vertices[face.indices[0]].point;
    let b = polytope.vertices[face.indices[1]].point;
    let c = polytope.vertices[face.indices[2]].point;

    // Project origin onto triangle to get barycentric coordinates.
    let (u, v, w) = barycentric_origin_projection(a, b, c);

    let sa = &polytope.vertices[face.indices[0]];
    let sb = &polytope.vertices[face.indices[1]];
    let sc = &polytope.vertices[face.indices[2]];

    let contact_a = Point3::from(
        sa.support_a.coords * u + sb.support_a.coords * v + sc.support_a.coords * w,
    );
    let contact_b = Point3::from(
        sa.support_b.coords * u + sb.support_b.coords * v + sc.support_b.coords * w,
    );

    (contact_a, contact_b)
}

/// Compute barycentric coordinates of the origin's projection onto triangle (a, b, c).
fn barycentric_origin_projection(a: Point3, b: Point3, c: Point3) -> (f64, f64, f64) {
    let ab = b - a;
    let ac = c - a;
    let ap = -a.coords; // origin - a

    let d00 = ab.dot(&ab);
    let d01 = ab.dot(&ac);
    let d11 = ac.dot(&ac);
    let d20 = ap.dot(&ab);
    let d21 = ap.dot(&ac);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < EPA_EPSILON {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    // Clamp to valid range.
    let u = u.max(0.0);
    let v = v.max(0.0);
    let w = w.max(0.0);
    let sum = u + v + w;
    (u / sum, v / sum, w / sum)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk::{gjk_intersection, GjkResult};
    use choreo_types::geometry::Sphere;

    fn at(x: f64, y: f64, z: f64) -> Transform3D {
        Transform3D::from_position(x, y, z)
    }

    fn unit_sphere() -> CollisionShape {
        CollisionShape::Sphere(Sphere::new([0.0, 0.0, 0.0], 1.0))
    }

    fn build_simplex(
        shape_a: &CollisionShape,
        ta: &Transform3D,
        shape_b: &CollisionShape,
        tb: &Transform3D,
    ) -> Option<Simplex> {
        // Run GJK then build simplex for EPA.
        let dirs = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];
        let mut simplex = Simplex::new();
        for dir in &dirs {
            let sp = minkowski_support(shape_a, ta, shape_b, tb, dir);
            simplex.push(sp);
        }
        Some(simplex)
    }

    #[test]
    fn test_epa_sphere_sphere_overlap() {
        let s = unit_sphere();
        let ta = at(0.0, 0.0, 0.0);
        let tb = at(1.0, 0.0, 0.0);

        let simplex = build_simplex(&s, &ta, &s, &tb).unwrap();
        let result = epa_penetration(&s, &ta, &s, &tb, &simplex);
        assert!(result.is_some());
        let r = result.unwrap();
        // Two unit spheres at distance 1.0: penetration ~1.0.
        assert!(r.penetration_depth > 0.0);
        assert!(r.penetration_depth < 2.5);
    }

    #[test]
    fn test_epa_sphere_sphere_deep_overlap() {
        let s = unit_sphere();
        let ta = at(0.0, 0.0, 0.0);
        let tb = at(0.5, 0.0, 0.0);

        let simplex = build_simplex(&s, &ta, &s, &tb).unwrap();
        let result = epa_penetration(&s, &ta, &s, &tb, &simplex);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.penetration_depth > 0.0);
    }

    #[test]
    fn test_epa_result_normal_direction() {
        let s = unit_sphere();
        let ta = at(0.0, 0.0, 0.0);
        let tb = at(0.8, 0.0, 0.0);

        let simplex = build_simplex(&s, &ta, &s, &tb).unwrap();
        let result = epa_penetration(&s, &ta, &s, &tb, &simplex);
        assert!(result.is_some());
        let r = result.unwrap();
        // Normal should roughly point along x-axis.
        assert!(r.penetration_normal.x.abs() > 0.5);
    }

    #[test]
    fn test_epa_polytope_from_simplex() {
        let s = unit_sphere();
        let ta = at(0.0, 0.0, 0.0);
        let tb = at(0.5, 0.0, 0.0);

        let simplex = build_simplex(&s, &ta, &s, &tb).unwrap();
        let polytope = EpaPolytope::from_simplex(&simplex);
        assert!(polytope.is_some());
        let p = polytope.unwrap();
        assert_eq!(p.vertices.len(), 4);
        assert!(p.faces.len() >= 4);
    }

    #[test]
    fn test_barycentric_coords() {
        let a = Point3::new(1.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        let c = Point3::new(0.0, 0.0, 1.0);
        let (u, v, w) = barycentric_origin_projection(a, b, c);
        assert!((u + v + w - 1.0).abs() < 1e-6);
        assert!(u >= 0.0 && v >= 0.0 && w >= 0.0);
    }

    #[test]
    fn test_epa_convergence() {
        // Test that EPA converges for various overlap amounts.
        let s = unit_sphere();
        for offset in &[0.1, 0.5, 1.0, 1.5, 1.9] {
            let ta = at(0.0, 0.0, 0.0);
            let tb = at(*offset, 0.0, 0.0);
            let simplex = build_simplex(&s, &ta, &s, &tb).unwrap();
            let result = epa_penetration(&s, &ta, &s, &tb, &simplex);
            assert!(result.is_some(), "EPA failed for offset {}", offset);
            let r = result.unwrap();
            assert!(r.penetration_depth > 0.0);
        }
    }
}
