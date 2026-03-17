//! Scene optimizer for simplifying, merging, and pruning scene elements.
//!
//! Provides optimization passes that reduce scene complexity while preserving
//! interaction semantics and spatial relationships.

use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use xr_types::geometry::{BoundingBox, Volume};
use xr_types::scene::{
    DependencyEdge, SceneModel,
};
use xr_types::ElementId;

/// Configuration for the scene optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Distance threshold for merging nearby elements of the same type.
    pub merge_distance: f64,
    /// Minimum volume below which elements are considered trivial.
    pub min_volume_threshold: f64,
    /// Maximum number of elements per BVH leaf node.
    pub bvh_leaf_size: usize,
    /// Whether to prune unreachable elements.
    pub prune_unreachable: bool,
    /// Whether to simplify composite volumes.
    pub simplify_volumes: bool,
    /// Whether to merge nearby same-type elements.
    pub merge_nearby: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            merge_distance: 0.05,
            min_volume_threshold: 1e-6,
            bvh_leaf_size: 4,
            prune_unreachable: true,
            simplify_volumes: true,
            merge_nearby: true,
        }
    }
}

/// Statistics about optimization results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub original_element_count: usize,
    pub final_element_count: usize,
    pub elements_merged: usize,
    pub elements_pruned: usize,
    pub volumes_simplified: usize,
    pub total_passes: usize,
}

impl OptimizationStats {
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_element_count == 0 {
            return 0.0;
        }
        1.0 - (self.final_element_count as f64 / self.original_element_count as f64)
    }
}

/// A BVH (Bounding Volume Hierarchy) node for accelerated spatial queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BvhNode {
    Leaf {
        bounds: BoundingBox,
        element_ids: Vec<ElementId>,
    },
    Internal {
        bounds: BoundingBox,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    pub fn bounds(&self) -> &BoundingBox {
        match self {
            BvhNode::Leaf { bounds, .. } => bounds,
            BvhNode::Internal { bounds, .. } => bounds,
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            BvhNode::Leaf { element_ids, .. } => element_ids.len(),
            BvhNode::Internal { left, right, .. } => {
                left.element_count() + right.element_count()
            }
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Internal { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Query all element IDs whose bounds intersect the given box.
    pub fn query(&self, query_box: &BoundingBox) -> Vec<ElementId> {
        let mut results = Vec::new();
        self.query_recursive(query_box, &mut results);
        results
    }

    fn query_recursive(&self, query_box: &BoundingBox, results: &mut Vec<ElementId>) {
        if !self.bounds().intersects(query_box) {
            return;
        }
        match self {
            BvhNode::Leaf { element_ids, .. } => {
                results.extend(element_ids.iter());
            }
            BvhNode::Internal { left, right, .. } => {
                left.query_recursive(query_box, results);
                right.query_recursive(query_box, results);
            }
        }
    }
}

/// The scene optimizer that applies various optimization passes.
#[derive(Debug)]
pub struct SceneOptimizer {
    config: OptimizerConfig,
}

impl SceneOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(OptimizerConfig::default())
    }

    /// Run all enabled optimization passes on a scene.
    pub fn optimize(&self, scene: &SceneModel) -> (SceneModel, OptimizationStats) {
        let mut stats = OptimizationStats {
            original_element_count: scene.elements.len(),
            ..Default::default()
        };
        let mut optimized = scene.clone();

        if self.config.simplify_volumes {
            let count = self.simplify_volumes_pass(&mut optimized);
            stats.volumes_simplified = count;
            stats.total_passes += 1;
        }

        if self.config.merge_nearby {
            let count = self.merge_nearby_pass(&mut optimized);
            stats.elements_merged = count;
            stats.total_passes += 1;
        }

        if self.config.prune_unreachable {
            let count = self.prune_unreachable_pass(&mut optimized);
            stats.elements_pruned = count;
            stats.total_passes += 1;
        }

        stats.final_element_count = optimized.elements.len();
        (optimized, stats)
    }

    /// Simplify composite volumes to their bounding boxes when they have many sub-volumes.
    fn simplify_volumes_pass(&self, scene: &mut SceneModel) -> usize {
        let mut count = 0;
        for elem in &mut scene.elements {
            if let Some(simplified) = self.simplify_volume(&elem.activation_volume) {
                elem.activation_volume = simplified;
                count += 1;
            }
        }
        count
    }

    fn simplify_volume(&self, vol: &Volume) -> Option<Volume> {
        match vol {
            Volume::Composite(children) if children.len() > 3 => {
                let bb = self.compute_volume_bounds(vol);
                Some(Volume::Box(bb))
            }
            Volume::Composite(children) => {
                let mut simplified_children = Vec::new();
                let mut any_changed = false;
                for child in children {
                    if let Some(s) = self.simplify_volume(child) {
                        simplified_children.push(s);
                        any_changed = true;
                    } else {
                        simplified_children.push(child.clone());
                    }
                }
                if any_changed {
                    Some(Volume::Composite(simplified_children))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn compute_volume_bounds(&self, vol: &Volume) -> BoundingBox {
        match vol {
            Volume::Box(bb) => *bb,
            Volume::Sphere(s) => {
                let r = s.radius;
                BoundingBox::from_center_extents(s.center, [r, r, r])
            }
            Volume::Capsule(c) => {
                let r = c.radius;
                let h = c.axis_length() / 2.0 + r;
                BoundingBox::from_center_extents(
                    [0.0, 0.0, 0.0],
                    [r, h, r],
                )
            }
            Volume::Cylinder(c) => {
                let r = c.radius;
                let h = c.half_height;
                BoundingBox::from_center_extents(
                    [0.0, 0.0, 0.0],
                    [r, h, r],
                )
            }
            Volume::ConvexHull(ch) => {
                if ch.vertices.is_empty() {
                    BoundingBox::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                } else {
                    let mut mn = ch.vertices[0];
                    let mut mx = ch.vertices[0];
                    for p in &ch.vertices[1..] {
                        mn[0] = mn[0].min(p[0]);
                        mn[1] = mn[1].min(p[1]);
                        mn[2] = mn[2].min(p[2]);
                        mx[0] = mx[0].max(p[0]);
                        mx[1] = mx[1].max(p[1]);
                        mx[2] = mx[2].max(p[2]);
                    }
                    let center = [
                        (mn[0] + mx[0]) / 2.0,
                        (mn[1] + mx[1]) / 2.0,
                        (mn[2] + mx[2]) / 2.0,
                    ];
                    let half = [
                        (mx[0] - mn[0]) / 2.0,
                        (mx[1] - mn[1]) / 2.0,
                        (mx[2] - mn[2]) / 2.0,
                    ];
                    BoundingBox::from_center_extents(center, half)
                }
            }
            Volume::Composite(children) => {
                if children.is_empty() {
                    BoundingBox::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                } else {
                    let first = self.compute_volume_bounds(&children[0]);
                    let mut result = first;
                    for child in &children[1..] {
                        let bb = self.compute_volume_bounds(child);
                        result = result.union(&bb);
                    }
                    result
                }
            }
        }
    }

    /// Merge nearby elements of the same interaction type.
    fn merge_nearby_pass(&self, scene: &mut SceneModel) -> usize {
        let mut merged_count = 0;
        let mut to_remove: HashSet<ElementId> = HashSet::new();
        let mut merge_map: HashMap<ElementId, Vec<ElementId>> = HashMap::new();

        // Group elements by interaction type
        let mut type_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, elem) in scene.elements.iter().enumerate() {
            let key = format!("{:?}", elem.interaction_type);
            type_groups.entry(key).or_default().push(i);
        }

        for (_type_key, indices) in &type_groups {
            if indices.len() < 2 {
                continue;
            }
            let mut used: HashSet<usize> = HashSet::new();
            for i in 0..indices.len() {
                if used.contains(&indices[i]) {
                    continue;
                }
                let mut cluster = vec![indices[i]];
                for j in (i + 1)..indices.len() {
                    if used.contains(&indices[j]) {
                        continue;
                    }
                    let a = scene.elements[indices[i]].position;
                    let b = scene.elements[indices[j]].position;
                    let dist = ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt();
                    if dist < self.config.merge_distance {
                        cluster.push(indices[j]);
                        used.insert(indices[j]);
                    }
                }
                if cluster.len() > 1 {
                    let primary_idx = cluster[0];
                    let primary_id = scene.elements[primary_idx].id;
                    for &idx in &cluster[1..] {
                        let id = scene.elements[idx].id;
                        to_remove.insert(id);
                        merge_map.entry(primary_id).or_default().push(id);
                        merged_count += 1;
                    }
                    // Expand primary element's bounds to cover merged elements
                    let positions: Vec<[f64; 3]> =
                        cluster.iter().map(|&idx| scene.elements[idx].position).collect();
                    let center = [
                        positions.iter().map(|p| p[0]).sum::<f64>() / positions.len() as f64,
                        positions.iter().map(|p| p[1]).sum::<f64>() / positions.len() as f64,
                        positions.iter().map(|p| p[2]).sum::<f64>() / positions.len() as f64,
                    ];
                    scene.elements[primary_idx].position = center;
                    // Create a bounding box that contains all merged elements
                    let max_dist = positions
                        .iter()
                        .map(|p| OrderedFloat(((p[0]-center[0]).powi(2) + (p[1]-center[1]).powi(2) + (p[2]-center[2]).powi(2)).sqrt()))
                        .max()
                        .map(|f| f.0)
                        .unwrap_or(0.05);
                    let half = max_dist + 0.05;
                    scene.elements[primary_idx].activation_volume =
                        Volume::Box(BoundingBox::from_center_extents(
                            center,
                            [half, half, half],
                        ));
                    let merged_names: Vec<String> =
                        cluster.iter().map(|&idx| scene.elements[idx].name.clone()).collect();
                    scene.elements[primary_idx].name =
                        format!("merged({})", merged_names.join("+"));
                }
            }
        }

        // Remove merged elements and remap dependencies
        if !to_remove.is_empty() {
            // Build id -> old index map
            let id_to_old_idx: HashMap<ElementId, usize> = scene.elements.iter().enumerate()
                .map(|(i, e)| (e.id, i)).collect();

            // Build merge redirect: old_index_of_merged -> old_index_of_primary
            let mut merge_redirect: HashMap<usize, usize> = HashMap::new();
            for (primary_id, merged_ids) in &merge_map {
                if let Some(&primary_idx) = id_to_old_idx.get(primary_id) {
                    for merged_id in merged_ids {
                        if let Some(&merged_idx) = id_to_old_idx.get(merged_id) {
                            merge_redirect.insert(merged_idx, primary_idx);
                        }
                    }
                }
            }

            // Build surviving mask and old->new index map
            let surviving: Vec<bool> = scene.elements.iter()
                .map(|e| !to_remove.contains(&e.id))
                .collect();
            let mut index_remap: HashMap<usize, usize> = HashMap::new();
            let mut new_idx = 0usize;
            for (old_idx, &survives) in surviving.iter().enumerate() {
                if survives {
                    index_remap.insert(old_idx, new_idx);
                    new_idx += 1;
                }
            }

            scene.elements.retain(|e| !to_remove.contains(&e.id));

            // Remap dependencies: redirect merged indices to primary, then remap
            for dep in &mut scene.dependencies {
                if let Some(&primary_idx) = merge_redirect.get(&dep.source_index) {
                    dep.source_index = primary_idx;
                }
                if let Some(&primary_idx) = merge_redirect.get(&dep.target_index) {
                    dep.target_index = primary_idx;
                }
                if let Some(&new_src) = index_remap.get(&dep.source_index) {
                    dep.source_index = new_src;
                }
                if let Some(&new_tgt) = index_remap.get(&dep.target_index) {
                    dep.target_index = new_tgt;
                }
            }
            // Remove self-loops
            scene.dependencies.retain(|d| d.source_index != d.target_index);
        }

        merged_count
    }

    /// Prune elements not reachable from any root element.
    fn prune_unreachable_pass(&self, scene: &mut SceneModel) -> usize {
        if scene.elements.is_empty() || scene.dependencies.is_empty() {
            return 0;
        }

        // Build adjacency for undirected reachability
        let element_ids: HashSet<ElementId> = scene.elements.iter().map(|e| e.id).collect();
        let mut adj: HashMap<ElementId, Vec<ElementId>> = HashMap::new();
        for dep in &scene.dependencies {
            if dep.source_index < scene.elements.len() && dep.target_index < scene.elements.len() {
                let from_id = scene.elements[dep.source_index].id;
                let to_id = scene.elements[dep.target_index].id;
                adj.entry(from_id).or_default().push(to_id);
                adj.entry(to_id).or_default().push(from_id);
            }
        }

        // Find elements that are part of any dependency
        let connected: HashSet<ElementId> = adj.keys().copied().collect();
        let isolated: HashSet<ElementId> = element_ids
            .difference(&connected)
            .copied()
            .collect();

        // Keep all connected elements + isolated ones (they might be standalone)
        // Only prune if an element has no volume and no deps (truly dead)
        let mut pruned = 0;
        let to_prune: HashSet<ElementId> = isolated
            .iter()
            .filter(|id| {
                let elem = scene.elements.iter().find(|e| e.id == **id);
                match elem {
                    Some(e) => compute_volume_size(&e.activation_volume) < self.config.min_volume_threshold,
                    None => false,
                }
            })
            .copied()
            .collect();

        if !to_prune.is_empty() {
            pruned = to_prune.len();
            let surviving: Vec<bool> = scene.elements.iter()
                .map(|e| !to_prune.contains(&e.id))
                .collect();
            scene.elements.retain(|e| !to_prune.contains(&e.id));
            Self::remap_deps_after_removal(&mut scene.dependencies, &surviving);
        }

        pruned
    }

    /// Build a BVH from the scene elements.
    pub fn build_bvh(&self, scene: &SceneModel) -> Option<BvhNode> {
        if scene.elements.is_empty() {
            return None;
        }

        let entries: Vec<(ElementId, BoundingBox)> = scene
            .elements
            .iter()
            .map(|elem| {
                let bb = self.compute_volume_bounds(&elem.activation_volume);
                (elem.id, bb)
            })
            .collect();

        Some(self.build_bvh_recursive(&entries))
    }

    fn build_bvh_recursive(&self, entries: &[(ElementId, BoundingBox)]) -> BvhNode {
        if entries.len() <= self.config.bvh_leaf_size {
            let bounds = entries
                .iter()
                .skip(1)
                .fold(entries[0].1, |acc, (_, bb)| acc.union(bb));
            return BvhNode::Leaf {
                bounds,
                element_ids: entries.iter().map(|(id, _)| *id).collect(),
            };
        }

        // Find the axis with greatest extent
        let overall_bounds = entries
            .iter()
            .skip(1)
            .fold(entries[0].1, |acc, (_, bb)| acc.union(bb));
        let extents = overall_bounds.extents();
        let split_axis = if extents[0] >= extents[1] && extents[0] >= extents[2] {
            0
        } else if extents[1] >= extents[2] {
            1
        } else {
            2
        };

        // Sort by the center of each bounding box along the split axis
        let mut sorted: Vec<(ElementId, BoundingBox)> = entries.to_vec();
        sorted.sort_by_key(|(_, bb)| {
            let c = bb.center();
            OrderedFloat(match split_axis {
                0 => c[0],
                1 => c[1],
                _ => c[2],
            })
        });

        let mid = sorted.len() / 2;
        let left = self.build_bvh_recursive(&sorted[..mid]);
        let right = self.build_bvh_recursive(&sorted[mid..]);

        let bounds = left.bounds().union(right.bounds());
        BvhNode::Internal {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Remove elements with volumes smaller than the threshold.
    pub fn remove_tiny_elements(&self, scene: &mut SceneModel) -> usize {
        let threshold = self.config.min_volume_threshold;
        let before = scene.elements.len();
        let removed_ids: HashSet<ElementId> = scene
            .elements
            .iter()
            .filter(|e| {
                compute_volume_size(&e.activation_volume) < threshold
            })
            .map(|e| e.id)
            .collect();
        let surviving: Vec<bool> = scene.elements.iter()
            .map(|e| !removed_ids.contains(&e.id))
            .collect();
        scene.elements.retain(|e| !removed_ids.contains(&e.id));
        Self::remap_deps_after_removal(&mut scene.dependencies, &surviving);
        before - scene.elements.len()
    }

    /// Deduplicate elements at the exact same position with the same type.
    pub fn deduplicate(&self, scene: &mut SceneModel) -> usize {
        let mut seen: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>, OrderedFloat<f64>, String), ElementId> =
            HashMap::new();
        let mut duplicates: HashSet<ElementId> = HashSet::new();

        for elem in &scene.elements {
            let key = (
                OrderedFloat(elem.position[0]),
                OrderedFloat(elem.position[1]),
                OrderedFloat(elem.position[2]),
                format!("{:?}", elem.interaction_type),
            );
            if let Some(_existing) = seen.get(&key) {
                duplicates.insert(elem.id);
            } else {
                seen.insert(key, elem.id);
            }
        }

        let count = duplicates.len();
        if count > 0 {
            let surviving: Vec<bool> = scene.elements.iter()
                .map(|e| !duplicates.contains(&e.id))
                .collect();
            scene.elements.retain(|e| !duplicates.contains(&e.id));
            Self::remap_deps_after_removal(&mut scene.dependencies, &surviving);
        }
        count
    }

    /// Sort elements by distance from a reference point (e.g., camera position).
    pub fn sort_by_distance(&self, scene: &mut SceneModel, reference: &[f64; 3]) {
        scene.elements.sort_by_key(|e| {
            let d = [e.position[0] - reference[0], e.position[1] - reference[1], e.position[2] - reference[2]];
            OrderedFloat((d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt())
        });
    }

    /// Remap dependency indices after removing elements.
    fn remap_deps_after_removal(deps: &mut Vec<DependencyEdge>, surviving: &[bool]) {
        let mut index_remap: HashMap<usize, usize> = HashMap::new();
        let mut new_idx = 0usize;
        for (old_idx, &survives) in surviving.iter().enumerate() {
            if survives {
                index_remap.insert(old_idx, new_idx);
                new_idx += 1;
            }
        }
        deps.retain(|d| {
            index_remap.contains_key(&d.source_index) && index_remap.contains_key(&d.target_index)
        });
        for dep in deps.iter_mut() {
            dep.source_index = index_remap[&dep.source_index];
            dep.target_index = index_remap[&dep.target_index];
        }
    }
}

/// Compute the approximate size of a volume.
fn compute_volume_size(vol: &Volume) -> f64 {
    match vol {
        Volume::Box(bb) => bb.volume(),
        Volume::Sphere(s) => (4.0 / 3.0) * std::f64::consts::PI * s.radius.powi(3),
        Volume::Capsule(c) => {
            std::f64::consts::PI * c.radius * c.radius * c.axis_length()
                + (4.0 / 3.0) * std::f64::consts::PI * c.radius.powi(3)
        }
        Volume::Cylinder(c) => std::f64::consts::PI * c.radius * c.radius * (2.0 * c.half_height),
        Volume::ConvexHull(_) => 0.0,
        Volume::Composite(vols) => vols.iter().map(|v| compute_volume_size(v)).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::Sphere;
    use xr_types::scene::{DependencyType, InteractableElement, InteractionType};

    fn make_scene() -> SceneModel {
        let mut scene = SceneModel::default();
        let mut btn = InteractableElement::new(
            "button1",
            [1.0, 0.0, 0.0],
            InteractionType::Click,
        );
        btn.activation_volume = Volume::Sphere(Sphere {
            center: [1.0, 0.0, 0.0],
            radius: 0.1,
        });
        scene.add_element(btn);

        let mut btn2 = InteractableElement::new(
            "button2",
            [1.01, 0.0, 0.0],
            InteractionType::Click,
        );
        btn2.activation_volume = Volume::Sphere(Sphere {
            center: [1.01, 0.0, 0.0],
            radius: 0.1,
        });
        scene.add_element(btn2);

        let slider = InteractableElement::new(
            "slider",
            [2.0, 0.0, 0.0],
            InteractionType::Slider,
        );
        scene.add_element(slider);

        let grab = InteractableElement::new(
            "handle",
            [0.0, 1.0, 0.0],
            InteractionType::Grab,
        );
        scene.add_element(grab);

        scene.add_dependency(0, 2, DependencyType::Sequential);
        scene
    }

    #[test]
    fn test_optimizer_defaults() {
        let optimizer = SceneOptimizer::with_defaults();
        assert!(optimizer.config.merge_distance > 0.0);
        assert!(optimizer.config.prune_unreachable);
    }

    #[test]
    fn test_optimize_merges_nearby() {
        let scene = make_scene();
        let optimizer = SceneOptimizer::new(OptimizerConfig {
            merge_distance: 0.1,
            ..Default::default()
        });
        let (optimized, stats) = optimizer.optimize(&scene);
        assert!(stats.elements_merged > 0);
        assert!(optimized.elements.len() < scene.elements.len());
    }

    #[test]
    fn test_simplify_composite_volumes() {
        let mut scene = SceneModel::default();
        let mut elem = InteractableElement::new(
            "complex",
            [0.0, 0.0, 0.0],
            InteractionType::Click,
        );
        elem.activation_volume = Volume::Composite(vec![
            Volume::Sphere(Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 0.1,
            }),
            Volume::Sphere(Sphere {
                center: [0.1, 0.0, 0.0],
                radius: 0.1,
            }),
            Volume::Sphere(Sphere {
                center: [0.2, 0.0, 0.0],
                radius: 0.1,
            }),
            Volume::Sphere(Sphere {
                center: [0.3, 0.0, 0.0],
                radius: 0.1,
            }),
        ]);
        scene.add_element(elem);

        let optimizer = SceneOptimizer::with_defaults();
        let (optimized, stats) = optimizer.optimize(&scene);
        assert_eq!(stats.volumes_simplified, 1);
        match &optimized.elements[0].activation_volume {
            Volume::Box(_) => {}
            other => panic!("Expected Box, got {:?}", other),
        }
    }

    #[test]
    fn test_build_bvh() {
        let scene = make_scene();
        let optimizer = SceneOptimizer::with_defaults();
        let bvh = optimizer.build_bvh(&scene);
        assert!(bvh.is_some());
        let bvh = bvh.unwrap();
        assert_eq!(bvh.element_count(), scene.elements.len());
        assert!(bvh.depth() >= 1);
    }

    #[test]
    fn test_bvh_query() {
        let scene = make_scene();
        let optimizer = SceneOptimizer::with_defaults();
        let bvh = optimizer.build_bvh(&scene).unwrap();
        let query_box = BoundingBox::from_center_extents(
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        );
        let results = bvh.query(&query_box);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_remove_tiny_elements() {
        let mut scene = SceneModel::default();
        let mut tiny = InteractableElement::new(
            "tiny",
            [0.0, 0.0, 0.0],
            InteractionType::Click,
        );
        tiny.activation_volume = Volume::Sphere(Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 0.0001,
        });
        scene.add_element(tiny);

        let mut normal = InteractableElement::new(
            "normal",
            [1.0, 0.0, 0.0],
            InteractionType::Click,
        );
        normal.activation_volume = Volume::Sphere(Sphere {
            center: [1.0, 0.0, 0.0],
            radius: 0.1,
        });
        scene.add_element(normal);

        let optimizer = SceneOptimizer::with_defaults();
        let removed = optimizer.remove_tiny_elements(&mut scene);
        assert_eq!(removed, 1);
        assert_eq!(scene.elements.len(), 1);
        assert_eq!(scene.elements[0].name, "normal");
    }

    #[test]
    fn test_deduplicate() {
        let mut scene = SceneModel::default();
        scene.add_element(InteractableElement::new(
            "btn1",
            [1.0, 0.0, 0.0],
            InteractionType::Click,
        ));
        scene.add_element(InteractableElement::new(
            "btn2",
            [1.0, 0.0, 0.0],
            InteractionType::Click,
        ));
        scene.add_element(InteractableElement::new(
            "slider",
            [2.0, 0.0, 0.0],
            InteractionType::Slider,
        ));

        let optimizer = SceneOptimizer::with_defaults();
        let deduped = optimizer.deduplicate(&mut scene);
        assert_eq!(deduped, 1);
        assert_eq!(scene.elements.len(), 2);
    }

    #[test]
    fn test_sort_by_distance() {
        let mut scene = make_scene();
        let optimizer = SceneOptimizer::with_defaults();
        let reference = [0.0, 0.0, 0.0];
        optimizer.sort_by_distance(&mut scene, &reference);
        let distances: Vec<f64> = scene
            .elements
            .iter()
            .map(|e| {
                let d = [e.position[0] - reference[0], e.position[1] - reference[1], e.position[2] - reference[2]];
                (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt()
            })
            .collect();
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_optimization_stats() {
        let scene = make_scene();
        let optimizer = SceneOptimizer::new(OptimizerConfig {
            merge_distance: 0.1,
            ..Default::default()
        });
        let (_optimized, stats) = optimizer.optimize(&scene);
        assert_eq!(stats.original_element_count, 4);
        assert!(stats.total_passes > 0);
        let ratio = stats.reduction_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_empty_scene_optimization() {
        let scene = SceneModel::default();
        let optimizer = SceneOptimizer::with_defaults();
        let (optimized, stats) = optimizer.optimize(&scene);
        assert_eq!(stats.original_element_count, 0);
        assert_eq!(stats.final_element_count, 0);
        assert_eq!(stats.reduction_ratio(), 0.0);
    }
}
