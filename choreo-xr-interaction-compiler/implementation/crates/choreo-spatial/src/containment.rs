//! Containment hierarchy and spatial subtyping.
//!
//! Provides a directed acyclic graph (DAG) for tracking region containment
//! relations, transitive-closure queries, containment layers for pruning,
//! and a spatial subtype checker using simplified LP feasibility.

use std::collections::{HashMap, HashSet, VecDeque};

use choreo_types::geometry::{AABB, Point3, Vector3};
use choreo_types::spatial::{RegionId, SpatialPredicateId, SpatialRegion};

// ─── Containment DAG ─────────────────────────────────────────────────────────

/// Directed acyclic graph of region containment relationships.
#[derive(Debug, Clone)]
pub struct ContainmentDAG {
    /// Edges: inner → set of outers (inner ⊂ outer).
    contained_by: HashMap<RegionId, HashSet<RegionId>>,
    /// Reverse edges: outer → set of inners.
    contains: HashMap<RegionId, HashSet<RegionId>>,
    /// All known regions.
    all_regions: HashSet<RegionId>,
}

impl ContainmentDAG {
    pub fn new() -> Self {
        Self {
            contained_by: HashMap::new(),
            contains: HashMap::new(),
            all_regions: HashSet::new(),
        }
    }

    /// Add a containment relationship: `inner` is contained in `outer`.
    pub fn add_containment(&mut self, inner: RegionId, outer: RegionId) {
        self.all_regions.insert(inner.clone());
        self.all_regions.insert(outer.clone());
        self.contained_by
            .entry(inner.clone())
            .or_default()
            .insert(outer.clone());
        self.contains
            .entry(outer)
            .or_default()
            .insert(inner);
    }

    /// Check if `inner` is (transitively) contained in `outer`.
    pub fn is_contained(&self, inner: &RegionId, outer: &RegionId) -> bool {
        if inner == outer {
            return true;
        }
        // BFS from inner through contained_by edges.
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(inner.clone());
        visited.insert(inner.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(outers) = self.contained_by.get(&current) {
                for o in outers {
                    if o == outer {
                        return true;
                    }
                    if visited.insert(o.clone()) {
                        queue.push_back(o.clone());
                    }
                }
            }
        }
        false
    }

    /// Get all regions that directly contain `region`.
    pub fn direct_containers(&self, region: &RegionId) -> Vec<&RegionId> {
        self.contained_by
            .get(region)
            .map(|s| s.iter().collect())
            .unwrap_or_default()
    }

    /// Get all regions directly contained in `region`.
    pub fn direct_contents(&self, region: &RegionId) -> Vec<&RegionId> {
        self.contains
            .get(region)
            .map(|s| s.iter().collect())
            .unwrap_or_default()
    }

    /// Get all regions transitively contained in `region`.
    pub fn transitive_contents(&self, region: &RegionId) -> HashSet<RegionId> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(region.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(inners) = self.contains.get(&current) {
                for inner in inners {
                    if result.insert(inner.clone()) {
                        queue.push_back(inner.clone());
                    }
                }
            }
        }
        result
    }

    /// Get all regions that transitively contain `region`.
    pub fn transitive_containers(&self, region: &RegionId) -> HashSet<RegionId> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(region.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(outers) = self.contained_by.get(&current) {
                for outer in outers {
                    if result.insert(outer.clone()) {
                        queue.push_back(outer.clone());
                    }
                }
            }
        }
        result
    }

    /// Compute containment layers (topological sort into depth levels).
    pub fn compute_containment_layers(&self) -> Vec<ContainmentLayer> {
        let roots = self.find_roots();
        if roots.is_empty() {
            return vec![];
        }

        let mut layers: Vec<ContainmentLayer> = Vec::new();
        let mut visited = HashSet::new();
        let mut current_level = roots;

        while !current_level.is_empty() {
            let layer = ContainmentLayer {
                depth: layers.len(),
                regions: current_level.clone(),
            };
            layers.push(layer);

            let mut next_level = Vec::new();
            for region in &current_level {
                visited.insert(region.clone());
                if let Some(inners) = self.contains.get(region) {
                    for inner in inners {
                        if !visited.contains(inner) {
                            // Only add if all containers are visited.
                            let all_containers_visited = self
                                .contained_by
                                .get(inner)
                                .map(|cs| cs.iter().all(|c| visited.contains(c)))
                                .unwrap_or(true);
                            if all_containers_visited {
                                next_level.push(inner.clone());
                            }
                        }
                    }
                }
            }
            current_level = next_level;
        }

        layers
    }

    /// Find root regions (not contained in any other region).
    fn find_roots(&self) -> Vec<RegionId> {
        self.all_regions
            .iter()
            .filter(|r| {
                self.contained_by
                    .get(*r)
                    .map(|s| s.is_empty())
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    /// All regions in the DAG.
    pub fn regions(&self) -> &HashSet<RegionId> {
        &self.all_regions
    }

    /// Number of regions.
    pub fn len(&self) -> usize {
        self.all_regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all_regions.is_empty()
    }
}

impl Default for ContainmentDAG {
    fn default() -> Self {
        Self::new()
    }
}

/// A layer of regions at the same depth in the containment hierarchy.
#[derive(Debug, Clone)]
pub struct ContainmentLayer {
    pub depth: usize,
    pub regions: Vec<RegionId>,
}

// ─── spatial subtype checker ─────────────────────────────────────────────────

/// Checks if one convex region is a spatial "subtype" of another,
/// meaning the inner region is completely contained in the outer region.
#[derive(Debug)]
pub struct SpatialSubtypeChecker {
    epsilon: f64,
}

impl SpatialSubtypeChecker {
    pub fn new() -> Self {
        Self { epsilon: 1e-6 }
    }

    pub fn with_epsilon(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Check if inner region is a subtype of (contained within) outer region.
    pub fn is_subtype(&self, inner: &SpatialRegion, outer: &SpatialRegion) -> bool {
        match (inner, outer) {
            (
                SpatialRegion::Sphere {
                    center: ci,
                    radius: ri,
                },
                SpatialRegion::Sphere {
                    center: co,
                    radius: ro,
                },
            ) => self.sphere_in_sphere(ci, *ri, co, *ro),
            (SpatialRegion::Aabb(inner_aabb), SpatialRegion::Aabb(outer_aabb)) => {
                self.aabb_in_aabb(inner_aabb, outer_aabb)
            }
            (
                SpatialRegion::Sphere {
                    center: ci,
                    radius: ri,
                },
                SpatialRegion::Aabb(outer_aabb),
            ) => self.sphere_in_aabb(ci, *ri, outer_aabb),
            (
                SpatialRegion::Aabb(inner_aabb),
                SpatialRegion::Sphere {
                    center: co,
                    radius: ro,
                },
            ) => self.aabb_in_sphere(inner_aabb, co, *ro),
            (SpatialRegion::ConvexHull { points: inner_pts }, SpatialRegion::Aabb(outer_aabb)) => {
                self.hull_points_in_aabb(inner_pts, outer_aabb)
            }
            (
                SpatialRegion::ConvexHull { points: inner_pts },
                SpatialRegion::Sphere {
                    center: co,
                    radius: ro,
                },
            ) => self.hull_points_in_sphere(inner_pts, co, *ro),
            (
                SpatialRegion::ConvexHull { points: inner_pts },
                SpatialRegion::ConvexHull { points: outer_pts },
            ) => self.polytope_in_polytope(inner_pts, outer_pts),
            _ => {
                // Fall back to AABB containment check.
                let inner_aabb = inner.to_aabb();
                let outer_aabb = outer.to_aabb();
                self.aabb_in_aabb(&inner_aabb, &outer_aabb)
            }
        }
    }

    fn sphere_in_sphere(
        &self,
        ci: &[f64; 3],
        ri: f64,
        co: &[f64; 3],
        ro: f64,
    ) -> bool {
        let dx = ci[0] - co[0];
        let dy = ci[1] - co[1];
        let dz = ci[2] - co[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        dist + ri <= ro + self.epsilon
    }

    fn aabb_in_aabb(&self, inner: &AABB, outer: &AABB) -> bool {
        inner.min[0] >= outer.min[0] - self.epsilon
            && inner.max[0] <= outer.max[0] + self.epsilon
            && inner.min[1] >= outer.min[1] - self.epsilon
            && inner.max[1] <= outer.max[1] + self.epsilon
            && inner.min[2] >= outer.min[2] - self.epsilon
            && inner.max[2] <= outer.max[2] + self.epsilon
    }

    fn sphere_in_aabb(&self, center: &[f64; 3], radius: f64, aabb: &AABB) -> bool {
        center[0] - radius >= aabb.min[0] - self.epsilon
            && center[0] + radius <= aabb.max[0] + self.epsilon
            && center[1] - radius >= aabb.min[1] - self.epsilon
            && center[1] + radius <= aabb.max[1] + self.epsilon
            && center[2] - radius >= aabb.min[2] - self.epsilon
            && center[2] + radius <= aabb.max[2] + self.epsilon
    }

    fn aabb_in_sphere(&self, aabb: &AABB, center: &[f64; 3], radius: f64) -> bool {
        // All 8 corners of the AABB must be inside the sphere.
        let corners = aabb.corners();
        let c = Point3::new(center[0], center[1], center[2]);
        let r_sq = (radius + self.epsilon) * (radius + self.epsilon);
        corners.iter().all(|corner| (corner - c).norm_squared() <= r_sq)
    }

    fn hull_points_in_aabb(&self, points: &[[f64; 3]], aabb: &AABB) -> bool {
        points.iter().all(|p| {
            p[0] >= aabb.min[0] - self.epsilon
                && p[0] <= aabb.max[0] + self.epsilon
                && p[1] >= aabb.min[1] - self.epsilon
                && p[1] <= aabb.max[1] + self.epsilon
                && p[2] >= aabb.min[2] - self.epsilon
                && p[2] <= aabb.max[2] + self.epsilon
        })
    }

    fn hull_points_in_sphere(&self, points: &[[f64; 3]], center: &[f64; 3], radius: f64) -> bool {
        let c = Point3::new(center[0], center[1], center[2]);
        let r_sq = (radius + self.epsilon) * (radius + self.epsilon);
        points.iter().all(|p| {
            let pt = Point3::new(p[0], p[1], p[2]);
            (pt - c).norm_squared() <= r_sq
        })
    }

    /// Convex polytope containment: check if all vertices of inner are inside
    /// outer. Uses a simplified approach: for each vertex of inner, verify it
    /// is on the correct side of all faces of the outer polytope's convex hull.
    ///
    /// This is equivalent to an LP feasibility check:
    ///   for each vertex v of inner:
    ///     for each face (normal n, offset d) of outer:
    ///       n·v <= d + epsilon
    fn polytope_in_polytope(
        &self,
        inner_pts: &[[f64; 3]],
        outer_pts: &[[f64; 3]],
    ) -> bool {
        if outer_pts.len() < 4 || inner_pts.is_empty() {
            return false;
        }

        // Compute face normals and offsets for the outer polytope.
        let faces = compute_convex_hull_faces(outer_pts);
        if faces.is_empty() {
            // Fallback to AABB.
            let outer_aabb = aabb_from_points(outer_pts);
            return inner_pts.iter().all(|p| {
                p[0] >= outer_aabb.min[0] - self.epsilon
                    && p[0] <= outer_aabb.max[0] + self.epsilon
                    && p[1] >= outer_aabb.min[1] - self.epsilon
                    && p[1] <= outer_aabb.max[1] + self.epsilon
                    && p[2] >= outer_aabb.min[2] - self.epsilon
                    && p[2] <= outer_aabb.max[2] + self.epsilon
            });
        }

        // Check each inner vertex against all outer faces (LP feasibility).
        for v in inner_pts {
            let pt = Vector3::new(v[0], v[1], v[2]);
            for (normal, offset) in &faces {
                if normal.dot(&pt) > *offset + self.epsilon {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for SpatialSubtypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute face half-planes (normal, offset) for a convex hull.
/// Each face is defined by n·x <= d.
fn compute_convex_hull_faces(points: &[[f64; 3]]) -> Vec<(Vector3, f64)> {
    if points.len() < 4 {
        return vec![];
    }

    let centroid = compute_centroid(points);
    let mut faces = Vec::new();

    // Generate faces from all triples of points (brute force for small sets).
    let n = points.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = Vector3::new(points[i][0], points[i][1], points[i][2]);
                let b = Vector3::new(points[j][0], points[j][1], points[j][2]);
                let c = Vector3::new(points[k][0], points[k][1], points[k][2]);

                let ab = b - a;
                let ac = c - a;
                let mut normal = ab.cross(&ac);
                let len = normal.norm();
                if len < 1e-10 {
                    continue;
                }
                normal /= len;

                let offset = normal.dot(&a);

                // Orient so centroid is on the "inside" (n·centroid <= d).
                let centroid_v = Vector3::new(centroid[0], centroid[1], centroid[2]);
                if normal.dot(&centroid_v) > offset {
                    normal = -normal;
                    let offset = -offset;
                    // Check all points are on the correct side.
                    let all_inside = points.iter().all(|p| {
                        let pv = Vector3::new(p[0], p[1], p[2]);
                        normal.dot(&pv) <= offset + 1e-6
                    });
                    if all_inside {
                        faces.push((normal, offset));
                    }
                } else {
                    let all_inside = points.iter().all(|p| {
                        let pv = Vector3::new(p[0], p[1], p[2]);
                        normal.dot(&pv) <= offset + 1e-6
                    });
                    if all_inside {
                        faces.push((normal, offset));
                    }
                }
            }
        }
    }

    faces
}

fn compute_centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mut c = [0.0; 3];
    for p in points {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c
}

fn aabb_from_points(points: &[[f64; 3]]) -> AABB {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    AABB::new(min, max)
}

// ─── feasible predicate set ──────────────────────────────────────────────────

/// Tracks which combinations of predicates are geometrically possible.
#[derive(Debug, Clone)]
pub struct FeasibleSet {
    /// Set of feasible predicate ID combinations.
    pub feasible_combinations: Vec<HashSet<SpatialPredicateId>>,
    /// Predicates known to be always true given the containment.
    pub always_true: HashSet<SpatialPredicateId>,
    /// Predicates known to be always false given the containment.
    pub always_false: HashSet<SpatialPredicateId>,
}

impl FeasibleSet {
    pub fn new() -> Self {
        Self {
            feasible_combinations: Vec::new(),
            always_true: HashSet::new(),
            always_false: HashSet::new(),
        }
    }

    /// Check if a specific predicate combination is feasible.
    pub fn is_feasible(&self, combination: &HashSet<SpatialPredicateId>) -> bool {
        if self.feasible_combinations.is_empty() {
            return true; // No constraints
        }
        self.feasible_combinations.iter().any(|fc| {
            combination.is_subset(fc) || fc.is_subset(combination)
        })
    }
}

impl Default for FeasibleSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the set of feasible predicate combinations given a containment DAG.
pub fn compute_feasible_predicate_set(
    dag: &ContainmentDAG,
    predicates: &[(SpatialPredicateId, choreo_types::spatial::SpatialPredicate)],
) -> FeasibleSet {
    let mut feasible = FeasibleSet::new();

    // Identify containment predicates.
    for (id, pred) in predicates {
        if let choreo_types::spatial::SpatialPredicate::Containment { inner, outer } = pred {
            if dag.is_contained(inner, outer) {
                feasible.always_true.insert(id.clone());
            }
        }
        if let choreo_types::spatial::SpatialPredicate::Inside { entity: _, region: _ } = pred {
            // If the entity's region is contained in the target region, always true.
            // This requires more context; skip for now.
        }
    }

    feasible
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(s: &str) -> RegionId {
        RegionId(s.to_string())
    }

    #[test]
    fn test_containment_dag_basic() {
        let mut dag = ContainmentDAG::new();
        dag.add_containment(rid("small"), rid("medium"));
        dag.add_containment(rid("medium"), rid("large"));

        assert!(dag.is_contained(&rid("small"), &rid("medium")));
        assert!(dag.is_contained(&rid("medium"), &rid("large")));
        assert!(dag.is_contained(&rid("small"), &rid("large"))); // transitive
        assert!(!dag.is_contained(&rid("large"), &rid("small")));
    }

    #[test]
    fn test_containment_dag_self() {
        let dag = ContainmentDAG::new();
        assert!(dag.is_contained(&rid("a"), &rid("a"))); // reflexive
    }

    #[test]
    fn test_containment_dag_layers() {
        let mut dag = ContainmentDAG::new();
        dag.add_containment(rid("a"), rid("root"));
        dag.add_containment(rid("b"), rid("root"));
        dag.add_containment(rid("c"), rid("a"));

        let layers = dag.compute_containment_layers();
        assert!(!layers.is_empty());
        // Root should be in the first layer.
        assert!(layers[0].regions.contains(&rid("root")));
    }

    #[test]
    fn test_transitive_contents() {
        let mut dag = ContainmentDAG::new();
        dag.add_containment(rid("a"), rid("b"));
        dag.add_containment(rid("b"), rid("c"));
        dag.add_containment(rid("d"), rid("c"));

        let contents = dag.transitive_contents(&rid("c"));
        assert!(contents.contains(&rid("b")));
        assert!(contents.contains(&rid("a")));
        assert!(contents.contains(&rid("d")));
    }

    #[test]
    fn test_transitive_containers() {
        let mut dag = ContainmentDAG::new();
        dag.add_containment(rid("a"), rid("b"));
        dag.add_containment(rid("b"), rid("c"));

        let containers = dag.transitive_containers(&rid("a"));
        assert!(containers.contains(&rid("b")));
        assert!(containers.contains(&rid("c")));
    }

    #[test]
    fn test_subtype_sphere_in_sphere() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let outer = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        assert!(checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_sphere_not_in_sphere() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Sphere {
            center: [4.0, 0.0, 0.0],
            radius: 2.0,
        };
        let outer = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        assert!(!checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_aabb_in_aabb() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Aabb(AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]));
        let outer = SpatialRegion::Aabb(AABB::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]));
        assert!(checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_aabb_not_in_aabb() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Aabb(AABB::new([-1.0, -1.0, -1.0], [6.0, 1.0, 1.0]));
        let outer = SpatialRegion::Aabb(AABB::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]));
        assert!(!checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_sphere_in_aabb() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let outer = SpatialRegion::Aabb(AABB::new([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
        assert!(checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_aabb_in_sphere() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::Aabb(AABB::new([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
        let outer = SpatialRegion::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };
        assert!(checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_subtype_hull_in_aabb() {
        let checker = SpatialSubtypeChecker::new();
        let inner = SpatialRegion::ConvexHull {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        let outer = SpatialRegion::Aabb(AABB::new([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]));
        assert!(checker.is_subtype(&inner, &outer));
    }

    #[test]
    fn test_feasible_set() {
        let mut dag = ContainmentDAG::new();
        dag.add_containment(rid("a"), rid("b"));

        let predicates = vec![(
            SpatialPredicateId("cont_ab".to_string()),
            choreo_types::spatial::SpatialPredicate::Containment {
                inner: rid("a"),
                outer: rid("b"),
            },
        )];

        let feasible = compute_feasible_predicate_set(&dag, &predicates);
        assert!(feasible.always_true.contains(&SpatialPredicateId("cont_ab".to_string())));
    }

    #[test]
    fn test_empty_dag() {
        let dag = ContainmentDAG::new();
        assert!(dag.is_empty());
        assert_eq!(dag.len(), 0);
        let layers = dag.compute_containment_layers();
        assert!(layers.is_empty());
    }
}
