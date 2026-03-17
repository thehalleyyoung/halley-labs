//! Bounding Volume Hierarchy (BVH).
//!
//! A binary BVH tree with surface area heuristic (SAH) construction, ray
//! traversal, frustum queries, sphere queries, refitting, and a flat/linear
//! layout for cache efficiency.

use choreo_types::geometry::{AABB, Point3, Vector3, Plane};

// ─── BVH node ────────────────────────────────────────────────────────────────

/// A node in the BVH.
#[derive(Debug, Clone)]
pub enum BVHNode<T> {
    Leaf {
        data: Vec<BVHItem<T>>,
        bounds: AABB,
    },
    Internal {
        left: Box<BVHNode<T>>,
        right: Box<BVHNode<T>>,
        bounds: AABB,
    },
}

/// An item stored in the BVH.
#[derive(Debug, Clone)]
pub struct BVHItem<T> {
    pub data: T,
    pub bounds: AABB,
}

impl<T> BVHItem<T> {
    pub fn new(data: T, bounds: AABB) -> Self {
        Self { data, bounds }
    }
}

impl<T: Clone> BVHNode<T> {
    fn bounds(&self) -> AABB {
        match self {
            BVHNode::Leaf { bounds, .. } => *bounds,
            BVHNode::Internal { bounds, .. } => *bounds,
        }
    }
}

// ─── ray hit ─────────────────────────────────────────────────────────────────

/// A hit from a ray traversal.
#[derive(Debug, Clone)]
pub struct Hit<'a, T> {
    pub t: f64,
    pub data: &'a T,
    pub bounds: AABB,
}

// ─── frustum ─────────────────────────────────────────────────────────────────

/// A view frustum defined by 6 planes (near, far, left, right, top, bottom).
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    pub fn new(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Test if an AABB is at least partially inside the frustum.
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let n = plane.normal_vec();
            // Find the p-vertex (the AABB corner most in the direction of the normal).
            let px = if n.x >= 0.0 { aabb.max[0] } else { aabb.min[0] };
            let py = if n.y >= 0.0 { aabb.max[1] } else { aabb.min[1] };
            let pz = if n.z >= 0.0 { aabb.max[2] } else { aabb.min[2] };
            let p_vertex = Point3::new(px, py, pz);
            if plane.signed_distance(&p_vertex) < 0.0 {
                return false;
            }
        }
        true
    }
}

// ─── BVH ─────────────────────────────────────────────────────────────────────

/// Bounding Volume Hierarchy.
pub struct BVH<T: Clone> {
    root: Option<BVHNode<T>>,
    size: usize,
    #[allow(dead_code)]
    max_leaf_size: usize,
}

impl<T: Clone + std::fmt::Debug> BVH<T> {
    /// Build a BVH from items using the Surface Area Heuristic.
    pub fn build(items: Vec<BVHItem<T>>) -> Self {
        Self::build_with_leaf_size(items, 4)
    }

    pub fn build_with_leaf_size(items: Vec<BVHItem<T>>, max_leaf_size: usize) -> Self {
        let size = items.len();
        if items.is_empty() {
            return Self {
                root: None,
                size: 0,
                max_leaf_size,
            };
        }
        let root = Self::build_recursive(items, max_leaf_size);
        Self {
            root: Some(root),
            size,
            max_leaf_size,
        }
    }

    fn build_recursive(items: Vec<BVHItem<T>>, max_leaf_size: usize) -> BVHNode<T> {
        if items.len() <= max_leaf_size {
            let bounds = items_bounds(&items);
            return BVHNode::Leaf {
                data: items,
                bounds,
            };
        }

        let total_bounds = items_bounds(&items);
        let (split_axis, split_pos) = find_sah_split(&items, &total_bounds);

        // Partition items.
        let (mut left_items, mut right_items): (Vec<_>, Vec<_>) = items
            .into_iter()
            .partition(|item| item.bounds.center()[split_axis] < split_pos);

        // If one side is empty, split in half.
        if left_items.is_empty() || right_items.is_empty() {
            let mut all = if left_items.is_empty() {
                right_items
            } else {
                left_items
            };
            all.sort_by(|a, b| {
                a.bounds.center()[split_axis]
                    .partial_cmp(&b.bounds.center()[split_axis])
                    .unwrap()
            });
            let mid = all.len() / 2;
            right_items = all.split_off(mid);
            left_items = all;
        }

        let left = Self::build_recursive(left_items, max_leaf_size);
        let right = Self::build_recursive(right_items, max_leaf_size);
        let bounds = left.bounds().merge(&right.bounds());

        BVHNode::Internal {
            left: Box::new(left),
            right: Box::new(right),
            bounds,
        }
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Root bounds.
    pub fn bounds(&self) -> Option<AABB> {
        self.root.as_ref().map(|r| r.bounds())
    }

    // ── ray traversal ────────────────────────────────────────────────────

    /// Traverse with a ray, returning all items whose AABB the ray hits.
    pub fn traverse_ray(&self, origin: &Point3, direction: &Vector3) -> Vec<Hit<'_, T>> {
        let mut hits = Vec::new();
        if let Some(ref root) = self.root {
            self.traverse_ray_recursive(root, origin, direction, &mut hits);
        }
        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        hits
    }

    fn traverse_ray_recursive<'a>(
        &'a self,
        node: &'a BVHNode<T>,
        origin: &Point3,
        direction: &Vector3,
        hits: &mut Vec<Hit<'a, T>>,
    ) {
        let node_bounds = node.bounds();
        if node_bounds.ray_intersect(origin, direction).is_none() {
            return;
        }

        match node {
            BVHNode::Leaf { data, bounds: _ } => {
                for item in data {
                    if let Some((t, _)) = item.bounds.ray_intersect(origin, direction) {
                        hits.push(Hit {
                            t,
                            data: &item.data,
                            bounds: item.bounds,
                        });
                    }
                }
            }
            BVHNode::Internal { left, right, .. } => {
                self.traverse_ray_recursive(left, origin, direction, hits);
                self.traverse_ray_recursive(right, origin, direction, hits);
            }
        }
    }

    // ── frustum traversal ────────────────────────────────────────────────

    /// Traverse with a frustum, returning all items at least partially inside.
    pub fn traverse_frustum(&self, frustum: &Frustum) -> Vec<&T> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.traverse_frustum_recursive(root, frustum, &mut results);
        }
        results
    }

    fn traverse_frustum_recursive<'a>(
        &'a self,
        node: &'a BVHNode<T>,
        frustum: &Frustum,
        results: &mut Vec<&'a T>,
    ) {
        if !frustum.intersects_aabb(&node.bounds()) {
            return;
        }

        match node {
            BVHNode::Leaf { data, .. } => {
                for item in data {
                    if frustum.intersects_aabb(&item.bounds) {
                        results.push(&item.data);
                    }
                }
            }
            BVHNode::Internal { left, right, .. } => {
                self.traverse_frustum_recursive(left, frustum, results);
                self.traverse_frustum_recursive(right, frustum, results);
            }
        }
    }

    // ── sphere traversal ─────────────────────────────────────────────────

    /// Traverse with a sphere, returning all items whose AABB overlaps the sphere.
    pub fn traverse_sphere(&self, center: &Point3, radius: f64) -> Vec<&T> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.traverse_sphere_recursive(root, center, radius, &mut results);
        }
        results
    }

    fn traverse_sphere_recursive<'a>(
        &'a self,
        node: &'a BVHNode<T>,
        center: &Point3,
        radius: f64,
        results: &mut Vec<&'a T>,
    ) {
        if !sphere_intersects_aabb(center, radius, &node.bounds()) {
            return;
        }

        match node {
            BVHNode::Leaf { data, .. } => {
                for item in data {
                    if sphere_intersects_aabb(center, radius, &item.bounds) {
                        results.push(&item.data);
                    }
                }
            }
            BVHNode::Internal { left, right, .. } => {
                self.traverse_sphere_recursive(left, center, radius, results);
                self.traverse_sphere_recursive(right, center, radius, results);
            }
        }
    }

    // ── refit ────────────────────────────────────────────────────────────

    /// Refit the BVH after items have moved. Requires mutable access to update bounds.
    pub fn refit<F>(&mut self, get_bounds: F)
    where
        F: Fn(&T) -> AABB,
    {
        if let Some(ref mut root) = self.root {
            Self::refit_recursive(root, &get_bounds);
        }
    }

    fn refit_recursive<F>(node: &mut BVHNode<T>, get_bounds: &F)
    where
        F: Fn(&T) -> AABB,
    {
        match node {
            BVHNode::Leaf { data, bounds } => {
                for item in data.iter_mut() {
                    item.bounds = get_bounds(&item.data);
                }
                *bounds = items_bounds(data);
            }
            BVHNode::Internal {
                left,
                right,
                bounds,
            } => {
                Self::refit_recursive(left, get_bounds);
                Self::refit_recursive(right, get_bounds);
                *bounds = left.bounds().merge(&right.bounds());
            }
        }
    }

    // ── flat/linear BVH ──────────────────────────────────────────────────

    /// Convert to a flat (linearized) BVH for cache-friendly traversal.
    pub fn to_linear(&self) -> LinearBVH<T>
    where
        T: Clone,
    {
        let mut nodes = Vec::new();
        let mut items = Vec::new();
        if let Some(ref root) = self.root {
            Self::linearize(root, &mut nodes, &mut items);
        }
        LinearBVH { nodes, items }
    }

    fn linearize(
        node: &BVHNode<T>,
        nodes: &mut Vec<LinearBVHNode>,
        items: &mut Vec<BVHItem<T>>,
    ) where
        T: Clone,
    {
        let idx = nodes.len();
        match node {
            BVHNode::Leaf { data, bounds } => {
                let item_start = items.len();
                items.extend(data.iter().cloned());
                nodes.push(LinearBVHNode {
                    bounds: *bounds,
                    kind: LinearBVHNodeKind::Leaf {
                        item_start,
                        item_count: data.len(),
                    },
                });
            }
            BVHNode::Internal {
                left,
                right,
                bounds,
            } => {
                // Reserve space for this node.
                nodes.push(LinearBVHNode {
                    bounds: *bounds,
                    kind: LinearBVHNodeKind::Internal {
                        left_child: 0,
                        right_child: 0,
                    },
                });

                let left_idx = nodes.len();
                Self::linearize(left, nodes, items);
                let right_idx = nodes.len();
                Self::linearize(right, nodes, items);

                if let LinearBVHNodeKind::Internal {
                    left_child,
                    right_child,
                } = &mut nodes[idx].kind
                {
                    *left_child = left_idx;
                    *right_child = right_idx;
                }
            }
        }
    }
}

/// Flat/linear BVH for cache-efficient traversal.
#[derive(Debug, Clone)]
pub struct LinearBVH<T> {
    pub nodes: Vec<LinearBVHNode>,
    pub items: Vec<BVHItem<T>>,
}

#[derive(Debug, Clone)]
pub struct LinearBVHNode {
    pub bounds: AABB,
    pub kind: LinearBVHNodeKind,
}

#[derive(Debug, Clone)]
pub enum LinearBVHNodeKind {
    Leaf {
        item_start: usize,
        item_count: usize,
    },
    Internal {
        left_child: usize,
        right_child: usize,
    },
}

impl<T: Clone + std::fmt::Debug> LinearBVH<T> {
    /// Traverse ray on the linear BVH.
    pub fn traverse_ray(&self, origin: &Point3, direction: &Vector3) -> Vec<(f64, &T)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        let mut results = Vec::new();
        let mut stack = vec![0usize];

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if node.bounds.ray_intersect(origin, direction).is_none() {
                continue;
            }

            match &node.kind {
                LinearBVHNodeKind::Leaf {
                    item_start,
                    item_count,
                } => {
                    for i in *item_start..(*item_start + *item_count) {
                        let item = &self.items[i];
                        if let Some((t, _)) = item.bounds.ray_intersect(origin, direction) {
                            results.push((t, &item.data));
                        }
                    }
                }
                LinearBVHNodeKind::Internal {
                    left_child,
                    right_child,
                } => {
                    stack.push(*right_child);
                    stack.push(*left_child);
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn items_bounds<T>(items: &[BVHItem<T>]) -> AABB {
    if items.is_empty() {
        return AABB::empty();
    }
    let mut bounds = items[0].bounds;
    for item in items.iter().skip(1) {
        bounds = bounds.merge(&item.bounds);
    }
    bounds
}

/// SAH split: find the best axis and position to split.
fn find_sah_split<T>(items: &[BVHItem<T>], total_bounds: &AABB) -> (usize, f64) {
    let n = items.len();
    if n <= 1 {
        return (0, total_bounds.center().x);
    }

    let mut best_cost = f64::INFINITY;
    let mut best_axis = 0;
    let mut best_pos = 0.0;

    let total_sa = total_bounds.surface_area();
    if total_sa < 1e-10 {
        return (0, total_bounds.center().x);
    }

    let num_bins = 12;

    for axis in 0..3 {
        let axis_min = total_bounds.min[axis];
        let axis_max = total_bounds.max[axis];
        if (axis_max - axis_min).abs() < 1e-10 {
            continue;
        }

        let bin_width = (axis_max - axis_min) / num_bins as f64;

        // Count items in each bin and compute bin bounds.
        let mut bin_counts = vec![0usize; num_bins];
        let mut bin_bounds = vec![AABB::empty(); num_bins];

        for item in items {
            let center = item.bounds.center()[axis];
            let bin = ((center - axis_min) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            bin_counts[bin] += 1;
            bin_bounds[bin] = if bin_bounds[bin].is_empty() {
                item.bounds
            } else {
                bin_bounds[bin].merge(&item.bounds)
            };
        }

        // Sweep from left.
        let mut left_count = 0;
        let mut left_bounds = AABB::empty();
        let mut left_areas = vec![0.0; num_bins];
        let mut left_counts = vec![0usize; num_bins];
        for i in 0..num_bins {
            left_count += bin_counts[i];
            left_bounds = if left_bounds.is_empty() && !bin_bounds[i].is_empty() {
                bin_bounds[i]
            } else if bin_bounds[i].is_empty() {
                left_bounds
            } else {
                left_bounds.merge(&bin_bounds[i])
            };
            left_areas[i] = if left_bounds.is_empty() {
                0.0
            } else {
                left_bounds.surface_area()
            };
            left_counts[i] = left_count;
        }

        // Sweep from right.
        let mut right_count = 0;
        let mut right_bounds = AABB::empty();
        let mut right_areas = vec![0.0; num_bins];
        let mut right_counts = vec![0usize; num_bins];
        for i in (0..num_bins).rev() {
            right_count += bin_counts[i];
            right_bounds = if right_bounds.is_empty() && !bin_bounds[i].is_empty() {
                bin_bounds[i]
            } else if bin_bounds[i].is_empty() {
                right_bounds
            } else {
                right_bounds.merge(&bin_bounds[i])
            };
            right_areas[i] = if right_bounds.is_empty() {
                0.0
            } else {
                right_bounds.surface_area()
            };
            right_counts[i] = right_count;
        }

        // Evaluate SAH cost at each split.
        for i in 0..(num_bins - 1) {
            let cost = left_counts[i] as f64 * left_areas[i]
                + right_counts[i + 1] as f64 * right_areas[i + 1];
            if cost < best_cost && left_counts[i] > 0 && right_counts[i + 1] > 0 {
                best_cost = cost;
                best_axis = axis;
                best_pos = axis_min + (i as f64 + 1.0) * bin_width;
            }
        }
    }

    (best_axis, best_pos)
}

fn sphere_intersects_aabb(center: &Point3, radius: f64, aabb: &AABB) -> bool {
    aabb.distance_to_point(center) <= radius
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u32, x: f64, y: f64, z: f64, s: f64) -> BVHItem<u32> {
        BVHItem::new(id, AABB::new([x - s, y - s, z - s], [x + s, y + s, z + s]))
    }

    #[test]
    fn test_build_empty() {
        let bvh: BVH<u32> = BVH::build(vec![]);
        assert!(bvh.is_empty());
    }

    #[test]
    fn test_build_single() {
        let items = vec![make_item(0, 0.0, 0.0, 0.0, 1.0)];
        let bvh = BVH::build(items);
        assert_eq!(bvh.len(), 1);
    }

    #[test]
    fn test_build_many() {
        let items: Vec<_> = (0..100)
            .map(|i| make_item(i, i as f64, 0.0, 0.0, 0.5))
            .collect();
        let bvh = BVH::build(items);
        assert_eq!(bvh.len(), 100);
    }

    #[test]
    fn test_ray_traversal() {
        let items: Vec<_> = (0..10)
            .map(|i| make_item(i, i as f64 * 3.0, 0.0, 0.0, 1.0))
            .collect();
        let bvh = BVH::build(items);

        let origin = Point3::new(-5.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let hits = bvh.traverse_ray(&origin, &direction);
        assert_eq!(hits.len(), 10);
        // Should be sorted by t.
        for i in 1..hits.len() {
            assert!(hits[i].t >= hits[i - 1].t);
        }
    }

    #[test]
    fn test_sphere_traversal() {
        let items: Vec<_> = (0..20)
            .map(|i| make_item(i, i as f64 * 2.0, 0.0, 0.0, 0.5))
            .collect();
        let bvh = BVH::build(items);

        let center = Point3::new(5.0, 0.0, 0.0);
        let results = bvh.traverse_sphere(&center, 3.0);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_frustum_traversal() {
        let items: Vec<_> = (0..10)
            .map(|i| make_item(i, i as f64, 0.0, 0.0, 0.5))
            .collect();
        let bvh = BVH::build(items);

        // Simple frustum that encompasses everything.
        let frustum = Frustum::new([
            Plane::new(Vector3::new(1.0, 0.0, 0.0), -100.0),  // left
            Plane::new(Vector3::new(-1.0, 0.0, 0.0), -100.0),  // right
            Plane::new(Vector3::new(0.0, 1.0, 0.0), -100.0),  // bottom
            Plane::new(Vector3::new(0.0, -1.0, 0.0), -100.0),  // top
            Plane::new(Vector3::new(0.0, 0.0, 1.0), -100.0),  // near
            Plane::new(Vector3::new(0.0, 0.0, -1.0), -100.0),  // far
        ]);
        let results = bvh.traverse_frustum(&frustum);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_refit() {
        let items: Vec<_> = (0..5)
            .map(|i| make_item(i, i as f64, 0.0, 0.0, 0.5))
            .collect();
        let mut bvh = BVH::build(items);
        let old_bounds = bvh.bounds().unwrap();

        // Refit with expanded bounds.
        bvh.refit(|_data| AABB::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]));
        let new_bounds = bvh.bounds().unwrap();
        assert!(new_bounds.volume() > old_bounds.volume());
    }

    #[test]
    fn test_linear_bvh() {
        let items: Vec<_> = (0..20)
            .map(|i| make_item(i, i as f64 * 2.0, 0.0, 0.0, 0.5))
            .collect();
        let bvh = BVH::build(items);
        let linear = bvh.to_linear();

        assert!(!linear.nodes.is_empty());
        assert!(!linear.items.is_empty());

        // Ray traversal on linear BVH should match tree BVH.
        let origin = Point3::new(-5.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let tree_hits = bvh.traverse_ray(&origin, &direction);
        let linear_hits = linear.traverse_ray(&origin, &direction);
        assert_eq!(tree_hits.len(), linear_hits.len());
    }

    #[test]
    fn test_ray_miss() {
        let items = vec![make_item(0, 0.0, 0.0, 0.0, 1.0)];
        let bvh = BVH::build(items);

        // Ray going away from the item.
        let origin = Point3::new(10.0, 10.0, 10.0);
        let direction = Vector3::new(1.0, 1.0, 1.0);
        let hits = bvh.traverse_ray(&origin, &direction);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_sphere_query_none() {
        let items = vec![make_item(0, 100.0, 100.0, 100.0, 1.0)];
        let bvh = BVH::build(items);

        let results = bvh.traverse_sphere(&Point3::new(0.0, 0.0, 0.0), 1.0);
        assert!(results.is_empty());
    }
}
