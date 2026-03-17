//! R-tree spatial index for efficient spatial queries.
//!
//! Provides `SpatialIndex` implementing an R-tree for bounding box queries,
//! range queries, k-nearest-neighbor, ray casting, and spatial joins.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use ordered_float::OrderedFloat;
use uuid::Uuid;

use xr_types::geometry::{BoundingBox, Ray, point_distance_sq};

/// An entry stored in the R-tree.
#[derive(Debug, Clone)]
pub struct RTreeEntry {
    pub id: Uuid,
    pub element_index: usize,
    pub bounds: BoundingBox,
    pub center: [f64; 3],
}

impl RTreeEntry {
    pub fn new(element_index: usize, bounds: BoundingBox) -> Self {
        Self {
            id: Uuid::new_v4(),
            element_index,
            bounds,
            center: bounds.center(),
        }
    }

    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }
}

/// Internal R-tree node.
#[derive(Debug, Clone)]
enum RTreeNode {
    Leaf {
        entries: Vec<RTreeEntry>,
    },
    Internal {
        children: Vec<RTreeChild>,
    },
}

#[derive(Debug, Clone)]
struct RTreeChild {
    bounds: BoundingBox,
    node: Box<RTreeNode>,
}

/// R-tree spatial index.
pub struct SpatialIndex {
    root: Option<RTreeNode>,
    count: usize,
    max_entries: usize,
    min_entries: usize,
}

impl SpatialIndex {
    pub fn new() -> Self {
        Self {
            root: None,
            count: 0,
            max_entries: 16,
            min_entries: 4,
        }
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        let min_entries = (max_entries / 4).max(2);
        Self {
            root: None,
            count: 0,
            max_entries,
            min_entries,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Insert an entry into the R-tree.
    pub fn insert(&mut self, entry: RTreeEntry) {
        self.count += 1;
        match self.root.take() {
            None => {
                self.root = Some(RTreeNode::Leaf {
                    entries: vec![entry],
                });
            }
            Some(root) => {
                let new_root = self.insert_into(root, entry);
                self.root = Some(new_root);
            }
        }
    }

    fn insert_into(&self, node: RTreeNode, entry: RTreeEntry) -> RTreeNode {
        match node {
            RTreeNode::Leaf { mut entries } => {
                entries.push(entry);
                if entries.len() <= self.max_entries {
                    RTreeNode::Leaf { entries }
                } else {
                    self.split_leaf(entries)
                }
            }
            RTreeNode::Internal { mut children } => {
                // Choose the child whose bounding box needs least enlargement
                let best_idx = self.choose_subtree(&children, &entry.bounds);
                let child = children.remove(best_idx);
                let new_node = self.insert_into(*child.node, entry);

                match new_node {
                    RTreeNode::Internal {
                        children: new_children,
                    } if new_children.len() > self.max_entries => {
                        // The child was split; add both halves
                        for c in new_children {
                            children.push(c);
                        }
                        if children.len() > self.max_entries {
                            self.split_internal(children)
                        } else {
                            RTreeNode::Internal { children }
                        }
                    }
                    other => {
                        let bounds = Self::node_bounds(&other);
                        children.insert(
                            best_idx,
                            RTreeChild {
                                bounds,
                                node: Box::new(other),
                            },
                        );
                        RTreeNode::Internal { children }
                    }
                }
            }
        }
    }

    fn choose_subtree(&self, children: &[RTreeChild], bounds: &BoundingBox) -> usize {
        let mut best_idx = 0;
        let mut best_enlargement = f64::INFINITY;
        let mut best_area = f64::INFINITY;

        for (i, child) in children.iter().enumerate() {
            let enlarged = child.bounds.union(bounds);
            let enlargement = enlarged.volume() - child.bounds.volume();
            if enlargement < best_enlargement
                || (enlargement == best_enlargement && child.bounds.volume() < best_area)
            {
                best_enlargement = enlargement;
                best_area = child.bounds.volume();
                best_idx = i;
            }
        }
        best_idx
    }

    fn split_leaf(&self, mut entries: Vec<RTreeEntry>) -> RTreeNode {
        // Quadratic split: pick the two entries whose combined bounding box wastes the most space
        let (seed1, seed2) = self.pick_seeds_entries(&entries);
        let e2 = entries.remove(seed2);
        let e1 = entries.remove(seed1);

        let mut group1 = vec![e1];
        let mut group2 = vec![e2];
        let mut bounds1 = group1[0].bounds;
        let mut bounds2 = group2[0].bounds;

        for entry in entries {
            if group1.len() + 1 == self.min_entries && group2.len() > self.min_entries {
                group1.push(entry);
                continue;
            }
            if group2.len() + 1 == self.min_entries && group1.len() > self.min_entries {
                group2.push(entry);
                continue;
            }

            let enlarge1 = bounds1.union(&entry.bounds).volume() - bounds1.volume();
            let enlarge2 = bounds2.union(&entry.bounds).volume() - bounds2.volume();
            if enlarge1 < enlarge2 {
                bounds1 = bounds1.union(&entry.bounds);
                group1.push(entry);
            } else {
                bounds2 = bounds2.union(&entry.bounds);
                group2.push(entry);
            }
        }

        let b1 = Self::entries_bounds(&group1);
        let b2 = Self::entries_bounds(&group2);

        RTreeNode::Internal {
            children: vec![
                RTreeChild {
                    bounds: b1,
                    node: Box::new(RTreeNode::Leaf { entries: group1 }),
                },
                RTreeChild {
                    bounds: b2,
                    node: Box::new(RTreeNode::Leaf { entries: group2 }),
                },
            ],
        }
    }

    fn split_internal(&self, mut children: Vec<RTreeChild>) -> RTreeNode {
        let (seed1, seed2) = self.pick_seeds_children(&children);
        let c2 = children.remove(seed2);
        let c1 = children.remove(seed1);

        let mut group1 = vec![c1];
        let mut group2 = vec![c2];
        let mut bounds1 = group1[0].bounds;
        let mut bounds2 = group2[0].bounds;

        for child in children {
            if group1.len() + 1 == self.min_entries && group2.len() > self.min_entries {
                group1.push(child);
                continue;
            }
            if group2.len() + 1 == self.min_entries && group1.len() > self.min_entries {
                group2.push(child);
                continue;
            }

            let enlarge1 = bounds1.union(&child.bounds).volume() - bounds1.volume();
            let enlarge2 = bounds2.union(&child.bounds).volume() - bounds2.volume();
            if enlarge1 < enlarge2 {
                bounds1 = bounds1.union(&child.bounds);
                group1.push(child);
            } else {
                bounds2 = bounds2.union(&child.bounds);
                group2.push(child);
            }
        }

        let b1 = Self::children_bounds(&group1);
        let b2 = Self::children_bounds(&group2);

        RTreeNode::Internal {
            children: vec![
                RTreeChild {
                    bounds: b1,
                    node: Box::new(RTreeNode::Internal { children: group1 }),
                },
                RTreeChild {
                    bounds: b2,
                    node: Box::new(RTreeNode::Internal { children: group2 }),
                },
            ],
        }
    }

    fn pick_seeds_entries(&self, entries: &[RTreeEntry]) -> (usize, usize) {
        let mut worst_waste = f64::NEG_INFINITY;
        let mut s1 = 0;
        let mut s2 = 1.min(entries.len().saturating_sub(1));
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let combined = entries[i].bounds.union(&entries[j].bounds);
                let waste =
                    combined.volume() - entries[i].bounds.volume() - entries[j].bounds.volume();
                if waste > worst_waste {
                    worst_waste = waste;
                    s1 = i;
                    s2 = j;
                }
            }
        }
        (s1, s2)
    }

    fn pick_seeds_children(&self, children: &[RTreeChild]) -> (usize, usize) {
        let mut worst_waste = f64::NEG_INFINITY;
        let mut s1 = 0;
        let mut s2 = 1.min(children.len().saturating_sub(1));
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                let combined = children[i].bounds.union(&children[j].bounds);
                let waste =
                    combined.volume() - children[i].bounds.volume() - children[j].bounds.volume();
                if waste > worst_waste {
                    worst_waste = waste;
                    s1 = i;
                    s2 = j;
                }
            }
        }
        (s1, s2)
    }

    fn entries_bounds(entries: &[RTreeEntry]) -> BoundingBox {
        let mut bounds = entries[0].bounds;
        for e in &entries[1..] {
            bounds = bounds.union(&e.bounds);
        }
        bounds
    }

    fn children_bounds(children: &[RTreeChild]) -> BoundingBox {
        let mut bounds = children[0].bounds;
        for c in &children[1..] {
            bounds = bounds.union(&c.bounds);
        }
        bounds
    }

    fn node_bounds(node: &RTreeNode) -> BoundingBox {
        match node {
            RTreeNode::Leaf { entries } => {
                if entries.is_empty() {
                    return BoundingBox::default();
                }
                Self::entries_bounds(entries)
            }
            RTreeNode::Internal { children } => {
                if children.is_empty() {
                    return BoundingBox::default();
                }
                Self::children_bounds(children)
            }
        }
    }

    /// Remove an entry by element index. Returns true if found.
    pub fn remove(&mut self, element_index: usize) -> bool {
        if let Some(root) = self.root.take() {
            let (new_root, found) = self.remove_from(root, element_index);
            self.root = Some(new_root);
            if found {
                self.count -= 1;
            }
            found
        } else {
            false
        }
    }

    fn remove_from(&self, node: RTreeNode, element_index: usize) -> (RTreeNode, bool) {
        match node {
            RTreeNode::Leaf { mut entries } => {
                let len_before = entries.len();
                entries.retain(|e| e.element_index != element_index);
                let found = entries.len() < len_before;
                (RTreeNode::Leaf { entries }, found)
            }
            RTreeNode::Internal { mut children } => {
                for child in &mut children {
                    if child.bounds.intersects(&BoundingBox::new(
                        [f64::NEG_INFINITY; 3],
                        [f64::INFINITY; 3],
                    )) {
                        let old_node = std::mem::replace(
                            &mut *child.node,
                            RTreeNode::Leaf {
                                entries: Vec::new(),
                            },
                        );
                        let (new_node, found) = self.remove_from(old_node, element_index);
                        child.bounds = Self::node_bounds(&new_node);
                        *child.node = new_node;
                        if found {
                            return (RTreeNode::Internal { children }, true);
                        }
                    }
                }
                (RTreeNode::Internal { children }, false)
            }
        }
    }

    /// Query all entries whose bounding box intersects the given box.
    pub fn query_range(&self, bounds: &BoundingBox) -> Vec<&RTreeEntry> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.query_range_node(root, bounds, &mut results);
        }
        results
    }

    fn query_range_node<'a>(
        &self,
        node: &'a RTreeNode,
        bounds: &BoundingBox,
        results: &mut Vec<&'a RTreeEntry>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.intersects(bounds) {
                        results.push(entry);
                    }
                }
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    if child.bounds.intersects(bounds) {
                        self.query_range_node(&child.node, bounds, results);
                    }
                }
            }
        }
    }

    /// Query all entries whose bounding box contains the given point.
    pub fn query_point(&self, point: &[f64; 3]) -> Vec<&RTreeEntry> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.query_point_node(root, point, &mut results);
        }
        results
    }

    fn query_point_node<'a>(
        &self,
        node: &'a RTreeNode,
        point: &[f64; 3],
        results: &mut Vec<&'a RTreeEntry>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    if entry.bounds.contains_point(point) {
                        results.push(entry);
                    }
                }
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    if child.bounds.contains_point(point) {
                        self.query_point_node(&child.node, point, results);
                    }
                }
            }
        }
    }

    /// K-nearest-neighbor search from a given point.
    pub fn knn(&self, point: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut heap: BinaryHeap<KnnCandidate> = BinaryHeap::new();
        if let Some(ref root) = self.root {
            self.knn_search(root, point, k, &mut heap);
        }

        let mut results: Vec<(usize, f64)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.element_index, c.dist.into_inner()))
            .collect();
        results.truncate(k);
        results
    }

    fn knn_search(
        &self,
        node: &RTreeNode,
        point: &[f64; 3],
        k: usize,
        heap: &mut BinaryHeap<KnnCandidate>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    let dist = min_dist_to_box(point, &entry.bounds);
                    let candidate = KnnCandidate {
                        element_index: entry.element_index,
                        dist: OrderedFloat(dist),
                    };
                    if heap.len() < k {
                        heap.push(candidate);
                    } else if let Some(top) = heap.peek() {
                        if dist < top.dist.into_inner() {
                            heap.pop();
                            heap.push(candidate);
                        }
                    }
                }
            }
            RTreeNode::Internal { children } => {
                let mut sorted_children: Vec<(usize, f64)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, min_dist_to_box(point, &c.bounds)))
                    .collect();
                sorted_children.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                for (i, min_d) in sorted_children {
                    // Prune: if min distance exceeds the current k-th nearest, skip
                    if heap.len() >= k {
                        if let Some(top) = heap.peek() {
                            if min_d > top.dist.into_inner() {
                                continue;
                            }
                        }
                    }
                    self.knn_search(&children[i].node, point, k, heap);
                }
            }
        }
    }

    /// Ray cast: find all entries whose bounding box is hit by the ray,
    /// sorted by distance.
    pub fn ray_cast(&self, ray: &Ray, max_dist: f64) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.ray_cast_node(root, ray, max_dist, &mut results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn ray_cast_node(
        &self,
        node: &RTreeNode,
        ray: &Ray,
        max_dist: f64,
        results: &mut Vec<(usize, f64)>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for entry in entries {
                    if let Some(t) = ray.intersects_bbox(&entry.bounds) {
                        if t <= max_dist {
                            results.push((entry.element_index, t));
                        }
                    }
                }
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    if let Some(t) = ray.intersects_bbox(&child.bounds) {
                        if t <= max_dist {
                            self.ray_cast_node(&child.node, ray, max_dist, results);
                        }
                    }
                }
            }
        }
    }

    /// Spatial join: find all pairs of entries from this index and another
    /// whose bounding boxes overlap.
    pub fn spatial_join(&self, other: &SpatialIndex) -> Vec<(usize, usize)> {
        let mut results = Vec::new();
        if let (Some(ref root_a), Some(ref root_b)) = (&self.root, &other.root) {
            self.spatial_join_nodes(root_a, root_b, &mut results);
        }
        results
    }

    fn spatial_join_nodes(
        &self,
        node_a: &RTreeNode,
        node_b: &RTreeNode,
        results: &mut Vec<(usize, usize)>,
    ) {
        match (node_a, node_b) {
            (RTreeNode::Leaf { entries: ea }, RTreeNode::Leaf { entries: eb }) => {
                for a in ea {
                    for b in eb {
                        if a.bounds.intersects(&b.bounds) {
                            results.push((a.element_index, b.element_index));
                        }
                    }
                }
            }
            (RTreeNode::Leaf { entries }, RTreeNode::Internal { children })
            | (RTreeNode::Internal { children }, RTreeNode::Leaf { entries }) => {
                let leaf = RTreeNode::Leaf {
                    entries: entries.clone(),
                };
                let leaf_bounds = SpatialIndex::node_bounds(&leaf);
                for child in children {
                    if child.bounds.intersects(&leaf_bounds) {
                        self.spatial_join_nodes(&leaf, &child.node, results);
                    }
                }
            }
            (RTreeNode::Internal { children: ca }, RTreeNode::Internal { children: cb }) => {
                for a in ca {
                    for b in cb {
                        if a.bounds.intersects(&b.bounds) {
                            self.spatial_join_nodes(&a.node, &b.node, results);
                        }
                    }
                }
            }
        }
    }

    /// Get the bounding box of the entire index.
    pub fn bounds(&self) -> Option<BoundingBox> {
        self.root.as_ref().map(|r| SpatialIndex::node_bounds(r))
    }

    /// Collect all entries in the index.
    pub fn all_entries(&self) -> Vec<&RTreeEntry> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.collect_entries(root, &mut results);
        }
        results
    }

    fn collect_entries<'a>(&self, node: &'a RTreeNode, results: &mut Vec<&'a RTreeEntry>) {
        match node {
            RTreeNode::Leaf { entries } => {
                results.extend(entries.iter());
            }
            RTreeNode::Internal { children } => {
                for child in children {
                    self.collect_entries(&child.node, results);
                }
            }
        }
    }

    /// Bulk load entries using Sort-Tile-Recursive (STR).
    pub fn bulk_load(mut entries: Vec<RTreeEntry>) -> Self {
        let count = entries.len();
        if entries.is_empty() {
            return Self::new();
        }

        let max_entries = 16;
        let root = Self::str_build(&mut entries, max_entries);
        Self {
            root: Some(root),
            count,
            max_entries,
            min_entries: 4,
        }
    }

    fn str_build(entries: &mut [RTreeEntry], max_entries: usize) -> RTreeNode {
        if entries.len() <= max_entries {
            return RTreeNode::Leaf {
                entries: entries.to_vec(),
            };
        }

        let n = entries.len();
        let num_slices = ((n as f64) / (max_entries as f64)).sqrt().ceil() as usize;
        let slice_size = (n + num_slices - 1) / num_slices;

        // Sort by x coordinate
        entries.sort_by(|a, b| {
            a.center[0]
                .partial_cmp(&b.center[0])
                .unwrap_or(Ordering::Equal)
        });

        let mut children = Vec::new();
        for x_slice in entries.chunks_mut(slice_size.max(1)) {
            // Sort each x-slice by y coordinate
            x_slice.sort_by(|a, b| {
                a.center[1]
                    .partial_cmp(&b.center[1])
                    .unwrap_or(Ordering::Equal)
            });

            for y_slice in x_slice.chunks_mut(max_entries.max(1)) {
                let child_node = Self::str_build(y_slice, max_entries);
                let bounds = Self::node_bounds(&child_node);
                children.push(RTreeChild {
                    bounds,
                    node: Box::new(child_node),
                });
            }
        }

        if children.len() == 1 {
            return *children.pop().unwrap().node;
        }

        RTreeNode::Internal { children }
    }

    /// Find all overlapping pairs within this index.
    pub fn find_overlapping_pairs(&self) -> Vec<(usize, usize)> {
        let entries = self.all_entries();
        let mut pairs = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                if entries[i].bounds.intersects(&entries[j].bounds) {
                    pairs.push((entries[i].element_index, entries[j].element_index));
                }
            }
        }
        pairs
    }

    /// Query entries within a radius of a point.
    pub fn query_radius(&self, center: &[f64; 3], radius: f64) -> Vec<&RTreeEntry> {
        let query_box = BoundingBox::from_center_extents(*center, [radius; 3]);
        let candidates = self.query_range(&query_box);
        let r_sq = radius * radius;
        candidates
            .into_iter()
            .filter(|e| {
                let ec = e.bounds.center();
                point_distance_sq(center, &ec) <= r_sq
            })
            .collect()
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimum distance from a point to a bounding box.
fn min_dist_to_box(point: &[f64; 3], bounds: &BoundingBox) -> f64 {
    let mut dist_sq = 0.0f64;
    for i in 0..3 {
        if point[i] < bounds.min[i] {
            let d = bounds.min[i] - point[i];
            dist_sq += d * d;
        } else if point[i] > bounds.max[i] {
            let d = point[i] - bounds.max[i];
            dist_sq += d * d;
        }
    }
    dist_sq.sqrt()
}

/// KNN candidate for the priority queue (max-heap by distance).
#[derive(Debug, Clone)]
struct KnnCandidate {
    element_index: usize,
    dist: OrderedFloat<f64>,
}

impl PartialEq for KnnCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for KnnCandidate {}
impl PartialOrd for KnnCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for KnnCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distances at top for easy removal
        self.dist.cmp(&other.dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(index: usize, x: f64, y: f64, z: f64, half: f64) -> RTreeEntry {
        RTreeEntry::new(
            index,
            BoundingBox::from_center_extents([x, y, z], [half, half, half]),
        )
    }

    #[test]
    fn test_insert_and_query() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 0.5));
        idx.insert(make_entry(1, 2.0, 0.0, 0.0, 0.5));
        idx.insert(make_entry(2, 0.0, 2.0, 0.0, 0.5));

        assert_eq!(idx.len(), 3);

        let query = BoundingBox::from_center_extents([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let results = idx.query_range(&query);
        assert!(results.iter().any(|e| e.element_index == 0));
    }

    #[test]
    fn test_query_point() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 1.0));
        idx.insert(make_entry(1, 5.0, 5.0, 5.0, 1.0));

        let results = idx.query_point(&[0.0, 0.0, 0.0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].element_index, 0);
    }

    #[test]
    fn test_knn() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 0.1));
        idx.insert(make_entry(1, 1.0, 0.0, 0.0, 0.1));
        idx.insert(make_entry(2, 2.0, 0.0, 0.0, 0.1));
        idx.insert(make_entry(3, 10.0, 0.0, 0.0, 0.1));

        let results = idx.knn(&[0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
    }

    #[test]
    fn test_ray_cast() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 5.0, 1.0));
        idx.insert(make_entry(1, 0.0, 0.0, 10.0, 1.0));
        idx.insert(make_entry(2, 5.0, 5.0, 5.0, 1.0));

        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let results = idx.ray_cast(&ray, 100.0);

        let hit_indices: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(hit_indices.contains(&0));
        assert!(hit_indices.contains(&1));
        assert!(!hit_indices.contains(&2));
    }

    #[test]
    fn test_remove() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 0.5));
        idx.insert(make_entry(1, 2.0, 0.0, 0.0, 0.5));
        assert_eq!(idx.len(), 2);

        assert!(idx.remove(0));
        assert_eq!(idx.len(), 1);
        assert!(!idx.remove(99));
    }

    #[test]
    fn test_bulk_load() {
        let entries: Vec<RTreeEntry> = (0..100)
            .map(|i| {
                let x = (i % 10) as f64;
                let y = (i / 10) as f64;
                make_entry(i, x, y, 0.0, 0.3)
            })
            .collect();

        let idx = SpatialIndex::bulk_load(entries);
        assert_eq!(idx.len(), 100);

        let query = BoundingBox::from_center_extents([5.0, 5.0, 0.0], [2.0, 2.0, 1.0]);
        let results = idx.query_range(&query);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_spatial_join() {
        let mut idx_a = SpatialIndex::new();
        idx_a.insert(make_entry(0, 0.0, 0.0, 0.0, 1.0));
        idx_a.insert(make_entry(1, 5.0, 5.0, 5.0, 1.0));

        let mut idx_b = SpatialIndex::new();
        idx_b.insert(make_entry(10, 0.5, 0.0, 0.0, 1.0));
        idx_b.insert(make_entry(11, 20.0, 20.0, 20.0, 1.0));

        let pairs = idx_a.spatial_join(&idx_b);
        assert!(pairs.iter().any(|&(a, b)| a == 0 && b == 10));
        assert!(!pairs.iter().any(|&(_, b)| b == 11));
    }

    #[test]
    fn test_overlapping_pairs() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 1.0));
        idx.insert(make_entry(1, 0.5, 0.0, 0.0, 1.0));
        idx.insert(make_entry(2, 10.0, 10.0, 10.0, 0.5));

        let pairs = idx.find_overlapping_pairs();
        assert!(pairs.contains(&(0, 1)));
        assert!(!pairs.iter().any(|&(a, b)| a == 0 && b == 2));
    }

    #[test]
    fn test_query_radius() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 0.1));
        idx.insert(make_entry(1, 1.0, 0.0, 0.0, 0.1));
        idx.insert(make_entry(2, 10.0, 0.0, 0.0, 0.1));

        let results = idx.query_radius(&[0.0, 0.0, 0.0], 2.0);
        assert!(results.iter().any(|e| e.element_index == 0));
        assert!(results.iter().any(|e| e.element_index == 1));
        assert!(!results.iter().any(|e| e.element_index == 2));
    }

    #[test]
    fn test_empty_index() {
        let idx = SpatialIndex::new();
        assert!(idx.is_empty());
        assert!(idx.bounds().is_none());
        assert!(idx.query_range(&BoundingBox::default()).is_empty());
        assert!(idx.knn(&[0.0; 3], 5).is_empty());
    }

    #[test]
    fn test_bounds() {
        let mut idx = SpatialIndex::new();
        idx.insert(make_entry(0, 0.0, 0.0, 0.0, 1.0));
        idx.insert(make_entry(1, 5.0, 0.0, 0.0, 1.0));
        let bounds = idx.bounds().unwrap();
        assert!(bounds.min[0] <= -1.0);
        assert!(bounds.max[0] >= 6.0);
    }

    #[test]
    fn test_many_insertions_and_splits() {
        let mut idx = SpatialIndex::with_capacity(4);
        for i in 0..50 {
            let x = (i as f64) * 0.5;
            idx.insert(make_entry(i, x, 0.0, 0.0, 0.2));
        }
        assert_eq!(idx.len(), 50);
        let all = idx.all_entries();
        assert_eq!(all.len(), 50);
    }
}
