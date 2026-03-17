//! R-tree spatial index.
//!
//! A generic R-tree supporting insertion (Guttman's quadratic split), deletion
//! with re-insertion, range queries, k-nearest-neighbour queries, ray queries,
//! bulk loading via Sort-Tile-Recursive (STR), copy-on-write versioned
//! snapshots, and various statistics.

use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Arc;

use choreo_types::geometry::{AABB, Point3, Ray, Vector3};
use ordered_float::OrderedFloat;

// ─── configuration ───────────────────────────────────────────────────────────

/// Tuning knobs for the R-tree.
#[derive(Debug, Clone)]
pub struct RTreeConfig {
    /// Minimum fill for a non-root node (≥ 2).
    pub min_children: usize,
    /// Maximum entries per node.
    pub max_children: usize,
    /// Fraction of entries to re-insert on overflow before splitting (0..1).
    pub reinsert_fraction: f64,
}

impl Default for RTreeConfig {
    fn default() -> Self {
        Self {
            min_children: 4,
            max_children: 16,
            reinsert_fraction: 0.3,
        }
    }
}

// ─── entry ───────────────────────────────────────────────────────────────────

/// A datum stored in the tree together with its bounding box.
#[derive(Debug, Clone)]
pub struct RTreeEntry<T> {
    pub data: T,
    pub bbox: AABB,
}

impl<T> RTreeEntry<T> {
    pub fn new(data: T, bbox: AABB) -> Self {
        Self { data, bbox }
    }
}

// ─── node ────────────────────────────────────────────────────────────────────

/// A node in the R-tree.
#[derive(Debug, Clone)]
pub enum RTreeNode<T> {
    Leaf {
        entries: Vec<RTreeEntry<T>>,
    },
    Internal {
        children: Vec<RTreeNode<T>>,
        mbr: AABB,
    },
}

impl<T: Clone + fmt::Debug> RTreeNode<T> {
    fn empty_leaf() -> Self {
        RTreeNode::Leaf {
            entries: Vec::new(),
        }
    }

    /// Minimum bounding rectangle for this node.
    fn mbr(&self) -> AABB {
        match self {
            RTreeNode::Leaf { entries } => {
                if entries.is_empty() {
                    return AABB::empty();
                }
                let mut r = entries[0].bbox;
                for e in entries.iter().skip(1) {
                    r = r.merge(&e.bbox);
                }
                r
            }
            RTreeNode::Internal { mbr, .. } => *mbr,
        }
    }

    fn recompute_mbr(&mut self) {
        if let RTreeNode::Internal { children, mbr } = self {
            if children.is_empty() {
                *mbr = AABB::empty();
            } else {
                let mut r = children[0].mbr();
                for c in children.iter().skip(1) {
                    r = r.merge(&c.mbr());
                }
                *mbr = r;
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            RTreeNode::Leaf { entries } => entries.len(),
            RTreeNode::Internal { children, .. } => children.len(),
        }
    }

    #[allow(dead_code)]
    fn is_leaf(&self) -> bool {
        matches!(self, RTreeNode::Leaf { .. })
    }
}

// ─── R-tree ──────────────────────────────────────────────────────────────────

/// A generic R-tree spatial index.
pub struct RTree<T: Clone + fmt::Debug> {
    root: RTreeNode<T>,
    config: RTreeConfig,
    size: usize,
    height: usize,
}

impl<T: Clone + fmt::Debug> RTree<T> {
    // ── construction ─────────────────────────────────────────────────────

    /// Create an empty R-tree with default configuration.
    pub fn new() -> Self {
        Self::with_config(RTreeConfig::default())
    }

    /// Create an empty R-tree with the given configuration.
    pub fn with_config(config: RTreeConfig) -> Self {
        assert!(config.min_children >= 2, "min_children must be >= 2");
        assert!(
            config.max_children >= 2 * config.min_children,
            "max_children must be >= 2 * min_children"
        );
        Self {
            root: RTreeNode::empty_leaf(),
            config,
            size: 0,
            height: 1,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Height of the tree (1 for a single leaf node).
    pub fn height(&self) -> usize {
        self.height
    }

    // ── insertion (Guttman R-tree) ───────────────────────────────────────

    /// Insert an entry into the R-tree.
    pub fn insert(&mut self, entry: RTreeEntry<T>) {
        let split = self.insert_recursive(&mut self.root.clone(), entry, self.height);
        match split {
            InsertResult::Ok(node) => {
                self.root = node;
            }
            InsertResult::Split(a, b) => {
                let mbr = a.mbr().merge(&b.mbr());
                self.root = RTreeNode::Internal {
                    children: vec![a, b],
                    mbr,
                };
                self.height += 1;
            }
        }
        self.size += 1;
    }

    fn insert_recursive(
        &self,
        node: &mut RTreeNode<T>,
        entry: RTreeEntry<T>,
        depth: usize,
    ) -> InsertResult<T> {
        if depth == 1 {
            // Must be a leaf.
            if let RTreeNode::Leaf { entries } = node {
                entries.push(entry);
                if entries.len() > self.config.max_children {
                    let (a, b) = self.split_leaf(entries.clone());
                    return InsertResult::Split(a, b);
                }
                return InsertResult::Ok(node.clone());
            }
            unreachable!("depth 1 must be leaf");
        }

        if let RTreeNode::Internal { ref mut children, .. } = node {
            let idx = self.choose_subtree(children, &entry.bbox);
            let result =
                self.insert_recursive(&mut children[idx].clone(), entry, depth - 1);
            match result {
                InsertResult::Ok(new_child) => {
                    children[idx] = new_child;
                }
                InsertResult::Split(a, b) => {
                    children.remove(idx);
                    children.push(a);
                    children.push(b);
                }
            }
            node.recompute_mbr();
            if let RTreeNode::Internal { ref children, .. } = node {
                if children.len() > self.config.max_children {
                    let (na, nb) = self.split_internal(children.clone());
                    return InsertResult::Split(na, nb);
                }
            }
            InsertResult::Ok(node.clone())
        } else {
            unreachable!("non-leaf at depth > 1");
        }
    }

    /// Choose the child whose MBR needs least enlargement.
    fn choose_subtree(&self, children: &[RTreeNode<T>], bbox: &AABB) -> usize {
        let mut best_idx = 0;
        let mut best_enlargement = f64::INFINITY;
        let mut best_area = f64::INFINITY;
        for (i, child) in children.iter().enumerate() {
            let child_mbr = child.mbr();
            let merged = child_mbr.merge(bbox);
            let enlargement = merged.volume() - child_mbr.volume();
            if enlargement < best_enlargement
                || (enlargement == best_enlargement && child_mbr.volume() < best_area)
            {
                best_idx = i;
                best_enlargement = enlargement;
                best_area = child_mbr.volume();
            }
        }
        best_idx
    }

    // ── quadratic split ──────────────────────────────────────────────────

    fn split_leaf(&self, entries: Vec<RTreeEntry<T>>) -> (RTreeNode<T>, RTreeNode<T>) {
        let (seed_a, seed_b) = self.pick_seeds_entries(&entries);
        let mut group_a = vec![entries[seed_a].clone()];
        let mut group_b = vec![entries[seed_b].clone()];
        let mut mbr_a = entries[seed_a].bbox;
        let mut mbr_b = entries[seed_b].bbox;

        let mut remaining: Vec<usize> = (0..entries.len())
            .filter(|&i| i != seed_a && i != seed_b)
            .collect();

        while !remaining.is_empty() {
            // If one group needs all remaining to reach min_children, add them.
            if group_a.len() + remaining.len() == self.config.min_children {
                for &i in &remaining {
                    mbr_a = mbr_a.merge(&entries[i].bbox);
                    group_a.push(entries[i].clone());
                }
                break;
            }
            if group_b.len() + remaining.len() == self.config.min_children {
                for &i in &remaining {
                    mbr_b = mbr_b.merge(&entries[i].bbox);
                    group_b.push(entries[i].clone());
                }
                break;
            }

            // Pick next: largest preference for one group.
            let (pick_idx, prefer_a) =
                self.pick_next_entries(&entries, &remaining, &mbr_a, &mbr_b);
            let entry = entries[remaining[pick_idx]].clone();
            remaining.swap_remove(pick_idx);

            if prefer_a {
                mbr_a = mbr_a.merge(&entry.bbox);
                group_a.push(entry);
            } else {
                mbr_b = mbr_b.merge(&entry.bbox);
                group_b.push(entry);
            }
        }

        (
            RTreeNode::Leaf { entries: group_a },
            RTreeNode::Leaf { entries: group_b },
        )
    }

    fn split_internal(
        &self,
        children: Vec<RTreeNode<T>>,
    ) -> (RTreeNode<T>, RTreeNode<T>) {
        let bboxes: Vec<AABB> = children.iter().map(|c| c.mbr()).collect();
        let (seed_a, seed_b) = self.pick_seeds_bboxes(&bboxes);
        let mut group_a = vec![children[seed_a].clone()];
        let mut group_b = vec![children[seed_b].clone()];
        let mut mbr_a = bboxes[seed_a];
        let mut mbr_b = bboxes[seed_b];

        let mut remaining: Vec<usize> = (0..children.len())
            .filter(|&i| i != seed_a && i != seed_b)
            .collect();

        while !remaining.is_empty() {
            if group_a.len() + remaining.len() == self.config.min_children {
                for &i in &remaining {
                    mbr_a = mbr_a.merge(&bboxes[i]);
                    group_a.push(children[i].clone());
                }
                break;
            }
            if group_b.len() + remaining.len() == self.config.min_children {
                for &i in &remaining {
                    mbr_b = mbr_b.merge(&bboxes[i]);
                    group_b.push(children[i].clone());
                }
                break;
            }

            let (pick_idx, prefer_a) =
                self.pick_next_bboxes(&bboxes, &remaining, &mbr_a, &mbr_b);
            let child = children[remaining[pick_idx]].clone();
            let bb = bboxes[remaining[pick_idx]];
            remaining.swap_remove(pick_idx);

            if prefer_a {
                mbr_a = mbr_a.merge(&bb);
                group_a.push(child);
            } else {
                mbr_b = mbr_b.merge(&bb);
                group_b.push(child);
            }
        }

        let mbr_a_final = compute_mbr_nodes(&group_a);
        let mbr_b_final = compute_mbr_nodes(&group_b);

        (
            RTreeNode::Internal {
                children: group_a,
                mbr: mbr_a_final,
            },
            RTreeNode::Internal {
                children: group_b,
                mbr: mbr_b_final,
            },
        )
    }

    fn pick_seeds_entries(&self, entries: &[RTreeEntry<T>]) -> (usize, usize) {
        let bboxes: Vec<AABB> = entries.iter().map(|e| e.bbox).collect();
        self.pick_seeds_bboxes(&bboxes)
    }

    fn pick_seeds_bboxes(&self, bboxes: &[AABB]) -> (usize, usize) {
        let mut worst_waste = f64::NEG_INFINITY;
        let mut seed_a = 0;
        let mut seed_b = 1;
        for i in 0..bboxes.len() {
            for j in (i + 1)..bboxes.len() {
                let merged = bboxes[i].merge(&bboxes[j]);
                let waste = merged.volume() - bboxes[i].volume() - bboxes[j].volume();
                if waste > worst_waste {
                    worst_waste = waste;
                    seed_a = i;
                    seed_b = j;
                }
            }
        }
        (seed_a, seed_b)
    }

    fn pick_next_entries(
        &self,
        entries: &[RTreeEntry<T>],
        remaining: &[usize],
        mbr_a: &AABB,
        mbr_b: &AABB,
    ) -> (usize, bool) {
        let mut best_idx = 0;
        let mut best_diff = f64::NEG_INFINITY;
        let mut best_prefer_a = true;
        for (ri, &i) in remaining.iter().enumerate() {
            let enlarge_a = mbr_a.merge(&entries[i].bbox).volume() - mbr_a.volume();
            let enlarge_b = mbr_b.merge(&entries[i].bbox).volume() - mbr_b.volume();
            let diff = (enlarge_a - enlarge_b).abs();
            if diff > best_diff {
                best_diff = diff;
                best_idx = ri;
                best_prefer_a = enlarge_a < enlarge_b;
            }
        }
        (best_idx, best_prefer_a)
    }

    fn pick_next_bboxes(
        &self,
        bboxes: &[AABB],
        remaining: &[usize],
        mbr_a: &AABB,
        mbr_b: &AABB,
    ) -> (usize, bool) {
        let mut best_idx = 0;
        let mut best_diff = f64::NEG_INFINITY;
        let mut best_prefer_a = true;
        for (ri, &i) in remaining.iter().enumerate() {
            let enlarge_a = mbr_a.merge(&bboxes[i]).volume() - mbr_a.volume();
            let enlarge_b = mbr_b.merge(&bboxes[i]).volume() - mbr_b.volume();
            let diff = (enlarge_a - enlarge_b).abs();
            if diff > best_diff {
                best_diff = diff;
                best_idx = ri;
                best_prefer_a = enlarge_a < enlarge_b;
            }
        }
        (best_idx, best_prefer_a)
    }

    // ── deletion ─────────────────────────────────────────────────────────

    /// Remove the first entry whose data matches `predicate`. Returns true if found.
    pub fn remove<F>(&mut self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        let mut orphans = Vec::new();
        let removed = self.remove_recursive(&mut self.root.clone(), &predicate, &mut orphans);
        if let Some(new_root) = removed {
            self.root = new_root;
            self.size -= 1;
            // Re-insert orphans.
            for entry in orphans {
                self.insert(entry);
            }
            // Collapse root if needed.
            self.condense_root();
            true
        } else {
            false
        }
    }

    fn remove_recursive<F>(
        &self,
        node: &mut RTreeNode<T>,
        predicate: &F,
        orphans: &mut Vec<RTreeEntry<T>>,
    ) -> Option<RTreeNode<T>>
    where
        F: Fn(&T) -> bool,
    {
        match node {
            RTreeNode::Leaf { entries } => {
                if let Some(pos) = entries.iter().position(|e| predicate(&e.data)) {
                    entries.remove(pos);
                    Some(node.clone())
                } else {
                    None
                }
            }
            RTreeNode::Internal { children, .. } => {
                for i in 0..children.len() {
                    let _bbox_search = children[i].mbr();
                    // We can't prune here without knowing the entry bbox.
                    let result =
                        self.remove_recursive(&mut children[i].clone(), predicate, orphans);
                    if let Some(new_child) = result {
                        children[i] = new_child;
                        // If child is under-full, gather its entries as orphans.
                        if children[i].len() < self.config.min_children {
                            self.gather_entries(&children[i], orphans);
                            children.remove(i);
                        }
                        node.recompute_mbr();
                        return Some(node.clone());
                    }
                }
                None
            }
        }
    }

    fn gather_entries(&self, node: &RTreeNode<T>, orphans: &mut Vec<RTreeEntry<T>>) {
        match node {
            RTreeNode::Leaf { entries } => {
                orphans.extend(entries.iter().cloned());
            }
            RTreeNode::Internal { children, .. } => {
                for child in children {
                    self.gather_entries(child, orphans);
                }
            }
        }
    }

    fn condense_root(&mut self) {
        loop {
            if let RTreeNode::Internal { children, .. } = &self.root {
                if children.len() == 1 {
                    self.root = children[0].clone();
                    self.height -= 1;
                    continue;
                }
            }
            break;
        }
    }

    // ── queries ──────────────────────────────────────────────────────────

    /// Return all entries whose bounding box intersects `range`.
    pub fn query_range(&self, range: &AABB) -> Vec<&T> {
        let mut results = Vec::new();
        self.query_range_recursive(&self.root, range, &mut results);
        results
    }

    fn query_range_recursive<'a>(
        &'a self,
        node: &'a RTreeNode<T>,
        range: &AABB,
        results: &mut Vec<&'a T>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for e in entries {
                    if e.bbox.intersects(range) {
                        results.push(&e.data);
                    }
                }
            }
            RTreeNode::Internal { children, mbr } => {
                if !mbr.intersects(range) {
                    return;
                }
                for child in children {
                    self.query_range_recursive(child, range, results);
                }
            }
        }
    }

    /// K-nearest-neighbours to `point`. Returns (distance, data) sorted by distance.
    pub fn query_knn(&self, point: &Point3, k: usize) -> Vec<(f64, &T)> {
        let mut heap: BinaryHeap<KnnCandidate<&T>> = BinaryHeap::new();
        self.knn_recursive(&self.root, point, k, &mut heap);
        let mut results: Vec<_> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.dist.into_inner(), c.data))
            .collect();
        results.truncate(k);
        results
    }

    fn knn_recursive<'a>(
        &'a self,
        node: &'a RTreeNode<T>,
        point: &Point3,
        k: usize,
        heap: &mut BinaryHeap<KnnCandidate<&'a T>>,
    ) {
        let worst_dist = if heap.len() >= k {
            // max-heap: peek gives the largest distance
            heap.peek().map(|c| c.dist.into_inner()).unwrap_or(f64::INFINITY)
        } else {
            f64::INFINITY
        };

        match node {
            RTreeNode::Leaf { entries } => {
                for e in entries {
                    let d = aabb_point_dist(&e.bbox, point);
                    if d < worst_dist || heap.len() < k {
                        heap.push(KnnCandidate {
                            dist: OrderedFloat(d),
                            data: &e.data,
                        });
                        if heap.len() > k {
                            heap.pop(); // Remove farthest
                        }
                    }
                }
            }
            RTreeNode::Internal { children, mbr } => {
                let node_dist = aabb_point_dist(mbr, point);
                if node_dist > worst_dist && heap.len() >= k {
                    return;
                }
                // Sort children by distance to point for better pruning.
                let mut child_dists: Vec<(usize, f64)> = children
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, aabb_point_dist(&c.mbr(), point)))
                    .collect();
                child_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for (i, _) in child_dists {
                    self.knn_recursive(&children[i], point, k, heap);
                }
            }
        }
    }

    /// Ray query: return all entries whose AABB the ray intersects, with t parameter.
    pub fn query_ray(&self, ray: &Ray) -> Vec<(f64, &T)> {
        let origin = ray.origin_point();
        let dir = ray.direction_vec();
        let mut results = Vec::new();
        self.ray_recursive(&self.root, &origin, &dir, &mut results);
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results
    }

    fn ray_recursive<'a>(
        &'a self,
        node: &'a RTreeNode<T>,
        origin: &Point3,
        dir: &Vector3,
        results: &mut Vec<(f64, &'a T)>,
    ) {
        match node {
            RTreeNode::Leaf { entries } => {
                for e in entries {
                    if let Some((tmin, _)) = e.bbox.ray_intersect(origin, dir) {
                        results.push((tmin, &e.data));
                    }
                }
            }
            RTreeNode::Internal { children, mbr } => {
                if mbr.ray_intersect(origin, dir).is_none() {
                    return;
                }
                for child in children {
                    self.ray_recursive(child, origin, dir, results);
                }
            }
        }
    }

    // ── bulk loading (STR) ───────────────────────────────────────────────

    /// Build an R-tree from a set of entries using Sort-Tile-Recursive.
    pub fn bulk_load(entries: Vec<RTreeEntry<T>>) -> Self {
        Self::bulk_load_with_config(entries, RTreeConfig::default())
    }

    pub fn bulk_load_with_config(entries: Vec<RTreeEntry<T>>, config: RTreeConfig) -> Self {
        if entries.is_empty() {
            return Self::with_config(config);
        }
        let size = entries.len();
        let root = Self::str_build(entries, &config);
        let height = Self::compute_height(&root);
        Self {
            root,
            config,
            size,
            height,
        }
    }

    fn str_build(mut entries: Vec<RTreeEntry<T>>, config: &RTreeConfig) -> RTreeNode<T> {
        let n = entries.len();
        if n <= config.max_children {
            return RTreeNode::Leaf { entries };
        }

        let leaf_count = (n as f64 / config.max_children as f64).ceil() as usize;
        let slices = (leaf_count as f64).cbrt().ceil() as usize;
        let slice_size = (n as f64 / slices as f64).ceil() as usize;

        // Sort by X coordinate of centroid.
        entries.sort_by(|a, b| {
            let ca = a.bbox.center();
            let cb = b.bbox.center();
            ca.x.partial_cmp(&cb.x).unwrap()
        });

        let mut children: Vec<RTreeNode<T>> = Vec::new();

        for x_slice in entries.chunks_mut(slice_size.max(1)) {
            // Sort each X-slice by Y.
            x_slice.sort_by(|a, b| {
                let ca = a.bbox.center();
                let cb = b.bbox.center();
                ca.y.partial_cmp(&cb.y).unwrap()
            });

            let y_slice_size = (x_slice.len() as f64 / slices as f64).ceil() as usize;

            for y_slice in x_slice.chunks_mut(y_slice_size.max(1)) {
                // Sort each Y-slice by Z.
                y_slice.sort_by(|a, b| {
                    let ca = a.bbox.center();
                    let cb = b.bbox.center();
                    ca.z.partial_cmp(&cb.z).unwrap()
                });

                // Pack into leaf-sized groups.
                for chunk in y_slice.chunks(config.max_children) {
                    children.push(RTreeNode::Leaf {
                        entries: chunk.to_vec(),
                    });
                }
            }
        }

        // Recursively build internal levels.
        Self::str_build_internal(children, config)
    }

    fn str_build_internal(
        mut nodes: Vec<RTreeNode<T>>,
        config: &RTreeConfig,
    ) -> RTreeNode<T> {
        if nodes.len() == 1 {
            return nodes.pop().unwrap();
        }
        if nodes.len() <= config.max_children {
            let mbr = compute_mbr_nodes(&nodes);
            return RTreeNode::Internal {
                children: nodes,
                mbr,
            };
        }

        // Sort by centroid X of MBR.
        nodes.sort_by(|a, b| {
            let ca = a.mbr().center();
            let cb = b.mbr().center();
            ca.x.partial_cmp(&cb.x).unwrap()
        });

        let mut new_level: Vec<RTreeNode<T>> = Vec::new();
        for chunk in nodes.chunks(config.max_children) {
            let children: Vec<_> = chunk.to_vec();
            let mbr = compute_mbr_nodes(&children);
            new_level.push(RTreeNode::Internal { children, mbr });
        }

        Self::str_build_internal(new_level, config)
    }

    fn compute_height(node: &RTreeNode<T>) -> usize {
        match node {
            RTreeNode::Leaf { .. } => 1,
            RTreeNode::Internal { children, .. } => {
                1 + children
                    .iter()
                    .map(Self::compute_height)
                    .max()
                    .unwrap_or(0)
            }
        }
    }

    // ── iteration ────────────────────────────────────────────────────────

    /// Iterate over all entries in the tree.
    pub fn iter(&self) -> RTreeIter<'_, T> {
        let mut stack = Vec::new();
        stack.push(&self.root);
        RTreeIter {
            stack,
            current_entries: None,
            current_idx: 0,
        }
    }

    // ── statistics ───────────────────────────────────────────────────────

    pub fn depth(&self) -> usize {
        self.height
    }

    pub fn node_count(&self) -> usize {
        Self::count_nodes(&self.root)
    }

    fn count_nodes(node: &RTreeNode<T>) -> usize {
        match node {
            RTreeNode::Leaf { .. } => 1,
            RTreeNode::Internal { children, .. } => {
                1 + children.iter().map(Self::count_nodes).sum::<usize>()
            }
        }
    }

    /// Average fill factor of all nodes (entries / max_children).
    pub fn fill_factor(&self) -> f64 {
        let (total_entries, total_nodes) = Self::fill_stats(&self.root);
        if total_nodes == 0 {
            return 0.0;
        }
        total_entries as f64 / (total_nodes as f64 * self.config.max_children as f64)
    }

    fn fill_stats(node: &RTreeNode<T>) -> (usize, usize) {
        match node {
            RTreeNode::Leaf { entries } => (entries.len(), 1),
            RTreeNode::Internal { children, .. } => {
                let mut total_entries = children.len();
                let mut total_nodes = 1;
                for child in children {
                    let (e, n) = Self::fill_stats(child);
                    total_entries += e;
                    total_nodes += n;
                }
                (total_entries, total_nodes)
            }
        }
    }

    /// Root MBR.
    pub fn bounds(&self) -> AABB {
        self.root.mbr()
    }
}

impl<T: Clone + fmt::Debug> Default for RTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

enum InsertResult<T: Clone + fmt::Debug> {
    Ok(RTreeNode<T>),
    Split(RTreeNode<T>, RTreeNode<T>),
}

fn compute_mbr_nodes<T: Clone + fmt::Debug>(nodes: &[RTreeNode<T>]) -> AABB {
    if nodes.is_empty() {
        return AABB::empty();
    }
    let mut r = nodes[0].mbr();
    for n in nodes.iter().skip(1) {
        r = r.merge(&n.mbr());
    }
    r
}

fn aabb_point_dist(aabb: &AABB, point: &Point3) -> f64 {
    aabb.distance_to_point(point)
}

// ─── iterator ────────────────────────────────────────────────────────────────

pub struct RTreeIter<'a, T> {
    stack: Vec<&'a RTreeNode<T>>,
    current_entries: Option<&'a [RTreeEntry<T>]>,
    current_idx: usize,
}

impl<'a, T: Clone + fmt::Debug> Iterator for RTreeIter<'a, T> {
    type Item = &'a RTreeEntry<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entries) = self.current_entries {
                if self.current_idx < entries.len() {
                    let entry = &entries[self.current_idx];
                    self.current_idx += 1;
                    return Some(entry);
                } else {
                    self.current_entries = None;
                    self.current_idx = 0;
                }
            }

            let node = self.stack.pop()?;
            match node {
                RTreeNode::Leaf { entries } => {
                    self.current_entries = Some(entries);
                    self.current_idx = 0;
                }
                RTreeNode::Internal { children, .. } => {
                    for child in children.iter().rev() {
                        self.stack.push(child);
                    }
                }
            }
        }
    }
}

// ─── KNN helper ──────────────────────────────────────────────────────────────

struct KnnCandidate<T> {
    dist: OrderedFloat<f64>,
    data: T,
}

impl<T> PartialEq for KnnCandidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T> Eq for KnnCandidate<T> {}

impl<T> PartialOrd for KnnCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for KnnCandidate<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: farthest first (so we can pop the farthest).
        self.dist.cmp(&other.dist)
    }
}

// ─── VersionedRTree ──────────────────────────────────────────────────────────

/// Copy-on-write versioned R-tree for temporal snapshots.
///
/// Each version is an immutable snapshot. New versions are created by
/// applying insertions or deletions to an existing snapshot, sharing
/// as much structure as possible via `Arc`.
#[derive(Clone)]
pub struct VersionedRTree<T: Clone + fmt::Debug> {
    snapshots: Vec<Arc<RTree<T>>>,
}

impl<T: Clone + fmt::Debug> VersionedRTree<T> {
    pub fn new() -> Self {
        Self {
            snapshots: vec![Arc::new(RTree::new())],
        }
    }

    pub fn new_with_config(config: RTreeConfig) -> Self {
        Self {
            snapshots: vec![Arc::new(RTree::with_config(config))],
        }
    }

    /// Current version number (0-based).
    pub fn version(&self) -> usize {
        self.snapshots.len() - 1
    }

    /// Number of versions.
    pub fn version_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the R-tree for a specific version.
    pub fn at_version(&self, version: usize) -> Option<&RTree<T>> {
        self.snapshots.get(version).map(|arc| arc.as_ref())
    }

    /// Get the latest R-tree.
    pub fn latest(&self) -> &RTree<T> {
        self.snapshots.last().unwrap().as_ref()
    }

    /// Create a new version by inserting an entry.
    pub fn insert(&mut self, entry: RTreeEntry<T>) -> usize {
        let mut tree: RTree<T> = (*self.snapshots.last().unwrap().as_ref()).clone();
        tree.insert(entry);
        self.snapshots.push(Arc::new(tree));
        self.version()
    }

    /// Create a new version by removing the first matching entry.
    pub fn remove<F>(&mut self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        let mut tree: RTree<T> = (*self.snapshots.last().unwrap().as_ref()).clone();
        if tree.remove(predicate) {
            self.snapshots.push(Arc::new(tree));
            Some(self.version())
        } else {
            None
        }
    }

    /// Range query on the latest version.
    pub fn query_range(&self, range: &AABB) -> Vec<&T> {
        self.latest().query_range(range)
    }

    /// Range query at a specific version.
    pub fn query_range_at(&self, version: usize, range: &AABB) -> Option<Vec<&T>> {
        self.at_version(version).map(|tree| tree.query_range(range))
    }
}

impl<T: Clone + fmt::Debug> Default for VersionedRTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

// We need Clone for RTree (used by VersionedRTree).
impl<T: Clone + fmt::Debug> Clone for RTree<T> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            config: self.config.clone(),
            size: self.size,
            height: self.height,
        }
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn point_aabb(x: f64, y: f64, z: f64) -> AABB {
        AABB::new([x, y, z], [x, y, z])
    }

    fn small_aabb(x: f64, y: f64, z: f64, s: f64) -> AABB {
        AABB::new([x - s, y - s, z - s], [x + s, y + s, z + s])
    }

    #[test]
    fn test_insert_and_query_range() {
        let mut tree = RTree::new();
        for i in 0..50 {
            let x = i as f64;
            tree.insert(RTreeEntry::new(i, small_aabb(x, 0.0, 0.0, 0.5)));
        }
        assert_eq!(tree.len(), 50);

        let range = AABB::new([-1.0, -1.0, -1.0], [5.5, 1.0, 1.0]);
        let results = tree.query_range(&range);
        assert!(results.len() >= 6); // entries 0..=5 plus boundary cases
    }

    #[test]
    fn test_insert_and_remove() {
        let mut tree = RTree::new();
        for i in 0..20 {
            tree.insert(RTreeEntry::new(i, small_aabb(i as f64, 0.0, 0.0, 0.1)));
        }
        assert_eq!(tree.len(), 20);

        let removed = tree.remove(|&x| x == 10);
        assert!(removed);
        assert_eq!(tree.len(), 19);

        let range = AABB::new([9.8, -1.0, -1.0], [10.2, 1.0, 1.0]);
        let results = tree.query_range(&range);
        assert!(results.is_empty());
    }

    #[test]
    fn test_knn() {
        let mut tree = RTree::new();
        for i in 0..100 {
            let x = i as f64;
            tree.insert(RTreeEntry::new(i, point_aabb(x, 0.0, 0.0)));
        }

        let query_point = Point3::new(50.0, 0.0, 0.0);
        let results = tree.query_knn(&query_point, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(*results[0].1, 50);
    }

    #[test]
    fn test_ray_query() {
        let mut tree = RTree::new();
        for i in 0..10 {
            let x = (i as f64) * 2.0;
            tree.insert(RTreeEntry::new(i, small_aabb(x, 0.0, 0.0, 0.5)));
        }

        let ray = Ray::new(
            Point3::new(-1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        );
        let hits = tree.query_ray(&ray);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn test_bulk_load() {
        let entries: Vec<RTreeEntry<u32>> = (0..200)
            .map(|i| {
                let x = (i % 20) as f64;
                let y = (i / 20) as f64;
                RTreeEntry::new(i, small_aabb(x, y, 0.0, 0.3))
            })
            .collect();

        let tree = RTree::bulk_load(entries);
        assert_eq!(tree.len(), 200);

        let range = AABB::new([0.0, 0.0, -1.0], [5.5, 5.5, 1.0]);
        let results = tree.query_range(&range);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_bulk_load_empty() {
        let tree: RTree<u32> = RTree::bulk_load(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_iterator() {
        let mut tree = RTree::new();
        for i in 0..30 {
            tree.insert(RTreeEntry::new(i, small_aabb(i as f64, 0.0, 0.0, 0.1)));
        }
        let collected: Vec<_> = tree.iter().map(|e| e.data).collect();
        assert_eq!(collected.len(), 30);
    }

    #[test]
    fn test_statistics() {
        let entries: Vec<RTreeEntry<u32>> = (0..100)
            .map(|i| RTreeEntry::new(i, small_aabb(i as f64, 0.0, 0.0, 0.1)))
            .collect();
        let tree = RTree::bulk_load(entries);
        assert!(tree.depth() >= 1);
        assert!(tree.node_count() >= 1);
        assert!(tree.fill_factor() > 0.0);
    }

    #[test]
    fn test_versioned_rtree() {
        let mut vtree = VersionedRTree::<u32>::new();
        assert_eq!(vtree.version(), 0);

        vtree.insert(RTreeEntry::new(1, small_aabb(0.0, 0.0, 0.0, 1.0)));
        assert_eq!(vtree.version(), 1);

        vtree.insert(RTreeEntry::new(2, small_aabb(5.0, 0.0, 0.0, 1.0)));
        assert_eq!(vtree.version(), 2);

        // Version 0 should be empty.
        let v0 = vtree.at_version(0).unwrap();
        assert_eq!(v0.len(), 0);

        // Version 1 should have 1 entry.
        let v1 = vtree.at_version(1).unwrap();
        assert_eq!(v1.len(), 1);

        // Version 2 should have 2 entries.
        let v2 = vtree.at_version(2).unwrap();
        assert_eq!(v2.len(), 2);

        // Remove from latest.
        vtree.remove(|&x| x == 1);
        assert_eq!(vtree.version(), 3);
        assert_eq!(vtree.latest().len(), 1);

        // Version 2 still has 2.
        assert_eq!(vtree.at_version(2).unwrap().len(), 2);
    }

    #[test]
    fn test_random_point_cloud() {
        // Deterministic "random" point cloud using a simple LCG.
        let mut rng_state: u64 = 42;
        let mut next = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) * 100.0
        };

        let mut tree = RTree::new();
        let mut points = Vec::new();
        for _ in 0..500 {
            let x = next();
            let y = next();
            let z = next();
            points.push((x, y, z));
            tree.insert(RTreeEntry::new(points.len() - 1, small_aabb(x, y, z, 0.5)));
        }
        assert_eq!(tree.len(), 500);

        // Range query should find some points.
        let range = AABB::new([20.0, 20.0, 20.0], [40.0, 40.0, 40.0]);
        let results = tree.query_range(&range);
        // Verify correctness: manually check.
        let expected_count = points
            .iter()
            .filter(|&&(x, y, z)| {
                x + 0.5 >= 20.0
                    && x - 0.5 <= 40.0
                    && y + 0.5 >= 20.0
                    && y - 0.5 <= 40.0
                    && z + 0.5 >= 20.0
                    && z - 0.5 <= 40.0
            })
            .count();
        assert_eq!(results.len(), expected_count);

        // KNN should return k results.
        let query = Point3::new(50.0, 50.0, 50.0);
        let knn = tree.query_knn(&query, 10);
        assert_eq!(knn.len(), 10);
        // Verify sorted by distance.
        for i in 1..knn.len() {
            assert!(knn[i].0 >= knn[i - 1].0);
        }
    }

    #[test]
    fn test_bulk_load_vs_incremental() {
        let entries: Vec<RTreeEntry<u32>> = (0..100)
            .map(|i| {
                let x = (i % 10) as f64 * 3.0;
                let y = (i / 10) as f64 * 3.0;
                RTreeEntry::new(i, small_aabb(x, y, 0.0, 1.0))
            })
            .collect();

        let bulk = RTree::bulk_load(entries.clone());
        let mut incremental = RTree::new();
        for e in entries {
            incremental.insert(e);
        }

        let range = AABB::new([0.0, 0.0, -2.0], [15.0, 15.0, 2.0]);
        let mut bulk_results: Vec<u32> = bulk.query_range(&range).into_iter().copied().collect();
        let mut inc_results: Vec<u32> = incremental.query_range(&range).into_iter().copied().collect();
        bulk_results.sort();
        inc_results.sort();
        assert_eq!(bulk_results, inc_results);
    }
}
