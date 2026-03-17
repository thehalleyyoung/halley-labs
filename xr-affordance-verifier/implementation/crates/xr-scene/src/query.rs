//! Scene query engine for spatial and graph-based queries.
//!
//! Provides a unified query interface over the scene graph, spatial index,
//! and transform hierarchy for complex scene interrogation.

use std::collections::{HashMap, HashSet, VecDeque};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use xr_types::geometry::{BoundingBox, Ray, Volume};
use xr_types::scene::{InteractableElement, InteractionType, SceneModel};
use xr_types::{ElementId, Point3, Vector3};

use crate::graph::SceneGraph;
use crate::spatial_index::{RTreeEntry, SpatialIndex};

/// Helper: convert `[f64; 3]` to `Point3`.
fn p3(a: [f64; 3]) -> Point3 {
    Point3::new(a[0], a[1], a[2])
}

/// Helper: distance between two `[f64; 3]` arrays.
fn arr_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Result of a spatial query, ranked by relevance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub element_id: ElementId,
    pub name: String,
    pub distance: f64,
    pub interaction_type: InteractionType,
    pub position: Point3,
}

impl QueryResult {
    pub fn new(
        element_id: ElementId,
        name: String,
        distance: f64,
        interaction_type: InteractionType,
        position: Point3,
    ) -> Self {
        Self {
            element_id,
            name,
            distance,
            interaction_type,
            position,
        }
    }
}

/// A frustum defined by position, direction and field of view for view-based queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frustum {
    pub near: f64,
    pub far: f64,
    pub fov_horizontal: f64,
    pub fov_vertical: f64,
    pub position: Point3,
    pub forward: Vector3,
    pub up: Vector3,
}

impl Frustum {
    pub fn new(
        position: Point3,
        forward: Vector3,
        up: Vector3,
        fov_horizontal: f64,
        fov_vertical: f64,
        near: f64,
        far: f64,
    ) -> Self {
        Self {
            near,
            far,
            fov_horizontal,
            fov_vertical,
            position,
            forward: forward.normalize(),
            up: up.normalize(),
        }
    }

    /// Check if a point is inside the frustum (approximate check).
    pub fn contains_point(&self, point: &Point3) -> bool {
        let to_point = point - self.position;
        let dist_forward = to_point.dot(&self.forward);
        if dist_forward < self.near || dist_forward > self.far {
            return false;
        }
        let right = self.forward.cross(&self.up).normalize();
        let dist_right = to_point.dot(&right);
        let half_width = dist_forward * (self.fov_horizontal / 2.0).tan();
        if dist_right.abs() > half_width {
            return false;
        }
        let dist_up = to_point.dot(&self.up);
        let half_height = dist_forward * (self.fov_vertical / 2.0).tan();
        if dist_up.abs() > half_height {
            return false;
        }
        true
    }

    /// Get the bounding box that encloses the frustum (conservative).
    pub fn bounding_box(&self) -> BoundingBox {
        let right = self.forward.cross(&self.up).normalize();
        let half_w_far = self.far * (self.fov_horizontal / 2.0).tan();
        let half_h_far = self.far * (self.fov_vertical / 2.0).tan();
        let center_far = self.position + self.forward * self.far;
        let corners = [
            center_far + right * half_w_far + self.up * half_h_far,
            center_far - right * half_w_far + self.up * half_h_far,
            center_far + right * half_w_far - self.up * half_h_far,
            center_far - right * half_w_far - self.up * half_h_far,
            self.position,
        ];
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;
        for c in &corners {
            min_x = min_x.min(c.x);
            min_y = min_y.min(c.y);
            min_z = min_z.min(c.z);
            max_x = max_x.max(c.x);
            max_y = max_y.max(c.y);
            max_z = max_z.max(c.z);
        }
        BoundingBox::from_center_extents(
            [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0],
            [(max_x - min_x) / 2.0, (max_y - min_y) / 2.0, (max_z - min_z) / 2.0],
        )
    }
}

/// Filter criteria for queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryFilter {
    pub interaction_types: Option<Vec<InteractionType>>,
    pub max_distance: Option<f64>,
    pub min_volume: Option<f64>,
    pub name_contains: Option<String>,
    pub exclude_ids: HashSet<ElementId>,
}

impl QueryFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_interaction_types(mut self, types: Vec<InteractionType>) -> Self {
        self.interaction_types = Some(types);
        self
    }

    pub fn with_max_distance(mut self, dist: f64) -> Self {
        self.max_distance = Some(dist);
        self
    }

    pub fn with_min_volume(mut self, vol: f64) -> Self {
        self.min_volume = Some(vol);
        self
    }

    pub fn with_name_contains(mut self, pattern: String) -> Self {
        self.name_contains = Some(pattern);
        self
    }

    pub fn excluding(mut self, ids: HashSet<ElementId>) -> Self {
        self.exclude_ids = ids;
        self
    }

    /// Check if an element passes this filter.
    pub fn matches(&self, element: &InteractableElement, distance: f64) -> bool {
        if self.exclude_ids.contains(&element.id) {
            return false;
        }
        if let Some(ref types) = self.interaction_types {
            if !types.contains(&element.interaction_type) {
                return false;
            }
        }
        if let Some(max_d) = self.max_distance {
            if distance > max_d {
                return false;
            }
        }
        if let Some(ref pat) = self.name_contains {
            if !element.name.to_lowercase().contains(&pat.to_lowercase()) {
                return false;
            }
        }
        if let Some(min_v) = self.min_volume {
            let v = volume_size(&element.activation_volume);
            if v < min_v {
                return false;
            }
        }
        true
    }
}

/// Compute approximate volume size.
fn volume_size(vol: &Volume) -> f64 {
    match vol {
        Volume::Box(bb) => bb.volume(),
        Volume::Sphere(s) => (4.0 / 3.0) * std::f64::consts::PI * s.radius.powi(3),
        Volume::Capsule(c) => {
            let h = c.axis_length();
            let cyl = std::f64::consts::PI * c.radius * c.radius * h;
            let sphere = (4.0 / 3.0) * std::f64::consts::PI * c.radius.powi(3);
            cyl + sphere
        }
        Volume::Cylinder(c) => {
            std::f64::consts::PI * c.radius * c.radius * 2.0 * c.half_height
        }
        Volume::ConvexHull(_) => 0.0,
        Volume::Composite(vols) => vols.iter().map(|v| volume_size(v)).sum(),
    }
}

/// Convert a Volume to a BoundingBox centered at a position.
fn volume_to_bb(vol: &Volume, pos: &[f64; 3]) -> BoundingBox {
    match vol {
        Volume::Box(bb) => *bb,
        Volume::Sphere(s) => {
            let r = s.radius;
            BoundingBox::from_center_extents(s.center, [r, r, r])
        }
        Volume::Capsule(c) => {
            let r = c.radius;
            let h = c.axis_length() / 2.0 + r;
            BoundingBox::from_center_extents(*pos, [r, h, r])
        }
        Volume::Cylinder(c) => {
            let r = c.radius;
            let h = c.half_height;
            BoundingBox::from_center_extents(*pos, [r, h, r])
        }
        Volume::ConvexHull(ch) => {
            if ch.vertices.is_empty() {
                BoundingBox::from_center_extents(*pos, [0.05, 0.05, 0.05])
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
                BoundingBox::from_center_extents(
                    [
                        (mn[0] + mx[0]) / 2.0,
                        (mn[1] + mx[1]) / 2.0,
                        (mn[2] + mx[2]) / 2.0,
                    ],
                    [
                        (mx[0] - mn[0]) / 2.0,
                        (mx[1] - mn[1]) / 2.0,
                        (mx[2] - mn[2]) / 2.0,
                    ],
                )
            }
        }
        Volume::Composite(vols) => {
            if vols.is_empty() {
                BoundingBox::from_center_extents(*pos, [0.05, 0.05, 0.05])
            } else {
                let first = volume_to_bb(&vols[0], pos);
                let mut result = first;
                for v in &vols[1..] {
                    let bb = volume_to_bb(v, pos);
                    result = result.union(&bb);
                }
                result
            }
        }
    }
}

/// The scene query engine for performing complex queries over scene data.
pub struct SceneQueryEngine {
    elements: Vec<InteractableElement>,
    element_map: HashMap<ElementId, usize>,
    spatial_index: SpatialIndex,
    graph: Option<SceneGraph>,
}

impl std::fmt::Debug for SceneQueryEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneQueryEngine")
            .field("element_count", &self.elements.len())
            .field("has_graph", &self.graph.is_some())
            .finish()
    }
}

impl SceneQueryEngine {
    /// Build a query engine from a scene model.
    pub fn from_scene(scene: &SceneModel) -> Self {
        let elements = scene.elements.clone();
        let mut element_map = HashMap::new();
        let mut spatial_index = SpatialIndex::new();
        for (i, elem) in elements.iter().enumerate() {
            element_map.insert(elem.id, i);
            let bb = volume_to_bb(&elem.activation_volume, &elem.position);
            spatial_index.insert(RTreeEntry::new(i, bb));
        }
        let graph = Some(SceneGraph::from_scene(scene));
        Self {
            elements,
            element_map,
            spatial_index,
            graph,
        }
    }

    /// Find elements by interaction type.
    pub fn find_by_type(&self, interaction_type: InteractionType) -> Vec<QueryResult> {
        let origin = [0.0f64; 3];
        self.elements
            .iter()
            .filter(|e| e.interaction_type == interaction_type)
            .map(|e| {
                let dist = arr_dist(e.position, origin);
                QueryResult::new(
                    e.id,
                    e.name.clone(),
                    dist,
                    e.interaction_type.clone(),
                    p3(e.position),
                )
            })
            .collect()
    }

    /// Find the k nearest elements to a point.
    pub fn find_nearest(&self, point: &Point3, k: usize) -> Vec<QueryResult> {
        let p = [point[0], point[1], point[2]];
        let neighbors = self.spatial_index.knn(&p, k);
        neighbors
            .into_iter()
            .filter_map(|(idx, dist)| {
                let elem = self.elements.get(idx)?;
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    dist,
                    elem.interaction_type.clone(),
                    p3(elem.position),
                ))
            })
            .collect()
    }

    /// Find elements within a given radius of a point.
    pub fn find_within_radius(&self, center: &Point3, radius: f64) -> Vec<QueryResult> {
        let c = [center[0], center[1], center[2]];
        let results = self.spatial_index.query_radius(&c, radius);
        results
            .into_iter()
            .filter_map(|entry| {
                let elem = self.elements.get(entry.element_index)?;
                let dist = arr_dist(elem.position, c);
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    dist,
                    elem.interaction_type.clone(),
                    p3(elem.position),
                ))
            })
            .collect()
    }

    /// Find elements within a bounding box.
    pub fn find_in_box(&self, bb: &BoundingBox) -> Vec<QueryResult> {
        let entries = self.spatial_index.query_range(bb);
        let center = bb.center();
        entries
            .into_iter()
            .filter_map(|entry| {
                let elem = self.elements.get(entry.element_index)?;
                let dist = arr_dist(elem.position, center);
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    dist,
                    elem.interaction_type.clone(),
                    p3(elem.position),
                ))
            })
            .collect()
    }

    /// Find elements visible within a frustum.
    pub fn frustum_query(&self, frustum: &Frustum) -> Vec<QueryResult> {
        let bb = frustum.bounding_box();
        let candidates = self.spatial_index.query_range(&bb);
        candidates
            .into_iter()
            .filter_map(|entry| {
                let elem = self.elements.get(entry.element_index)?;
                let pos = p3(elem.position);
                if !frustum.contains_point(&pos) {
                    return None;
                }
                let dist = (pos - frustum.position).norm();
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    dist,
                    elem.interaction_type.clone(),
                    pos,
                ))
            })
            .collect()
    }

    /// Find elements along a ray within a max distance.
    pub fn ray_query(&self, ray: &Ray, max_distance: f64, tolerance: f64) -> Vec<QueryResult> {
        let hits = self.spatial_index.ray_cast(ray, max_distance);
        hits.into_iter()
            .filter_map(|(idx, t)| {
                let elem = self.elements.get(idx)?;
                let hit_point = [
                    ray.origin[0] + ray.direction[0] * t,
                    ray.origin[1] + ray.direction[1] * t,
                    ray.origin[2] + ray.direction[2] * t,
                ];
                let elem_dist = arr_dist(elem.position, hit_point);
                if elem_dist > tolerance {
                    return None;
                }
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    t,
                    elem.interaction_type.clone(),
                    p3(elem.position),
                ))
            })
            .collect()
    }

    /// Advanced filtered query combining spatial and attribute filters.
    pub fn filtered_query(
        &self,
        center: &Point3,
        radius: f64,
        filter: &QueryFilter,
    ) -> Vec<QueryResult> {
        let c = [center[0], center[1], center[2]];
        let candidates = self.spatial_index.query_radius(&c, radius);
        let mut results: Vec<QueryResult> = candidates
            .into_iter()
            .filter_map(|entry| {
                let elem = self.elements.get(entry.element_index)?;
                let dist = arr_dist(elem.position, c);
                if !filter.matches(elem, dist) {
                    return None;
                }
                Some(QueryResult::new(
                    elem.id,
                    elem.name.clone(),
                    dist,
                    elem.interaction_type.clone(),
                    p3(elem.position),
                ))
            })
            .collect();
        results.sort_by_key(|r| OrderedFloat(r.distance));
        results
    }

    /// Find all elements reachable from a given element via dependencies.
    pub fn find_reachable(&self, from: ElementId) -> Vec<ElementId> {
        if let Some(ref graph) = self.graph {
            if let Some(&start_idx) = self.element_map.get(&from) {
                let mut visited = HashSet::new();
                let mut queue = VecDeque::new();
                queue.push_back(start_idx);
                visited.insert(start_idx);
                while let Some(current) = queue.pop_front() {
                    for succ in graph.successors(current) {
                        if visited.insert(succ) {
                            queue.push_back(succ);
                        }
                    }
                }
                visited.remove(&start_idx);
                visited
                    .into_iter()
                    .filter_map(|idx| self.elements.get(idx).map(|e| e.id))
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }

    /// Find dependency paths between two elements.
    pub fn find_paths(&self, from: ElementId, to: ElementId) -> Vec<Vec<ElementId>> {
        if let Some(ref graph) = self.graph {
            if let (Some(&from_idx), Some(&to_idx)) =
                (self.element_map.get(&from), self.element_map.get(&to))
            {
                let paths = graph.all_paths(from_idx, to_idx, 10);
                paths
                    .into_iter()
                    .map(|path| {
                        path.into_iter()
                            .filter_map(|idx| self.elements.get(idx).map(|e| e.id))
                            .collect()
                    })
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }

    /// Get elements grouped by interaction type.
    pub fn group_by_type(&self) -> HashMap<String, Vec<QueryResult>> {
        let origin = [0.0f64; 3];
        let mut groups: HashMap<String, Vec<QueryResult>> = HashMap::new();
        for elem in &self.elements {
            let key = format!("{:?}", elem.interaction_type);
            let dist = arr_dist(elem.position, origin);
            let qr = QueryResult::new(
                elem.id,
                elem.name.clone(),
                dist,
                elem.interaction_type.clone(),
                p3(elem.position),
            );
            groups.entry(key).or_default().push(qr);
        }
        groups
    }

    /// Get density map: count of elements per spatial cell.
    pub fn density_map(&self, cell_size: f64) -> HashMap<(i64, i64, i64), usize> {
        let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
        for elem in &self.elements {
            let cx = (elem.position[0] / cell_size).floor() as i64;
            let cy = (elem.position[1] / cell_size).floor() as i64;
            let cz = (elem.position[2] / cell_size).floor() as i64;
            *map.entry((cx, cy, cz)).or_insert(0) += 1;
        }
        map
    }

    /// Get element count.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Get a specific element by ID.
    pub fn get_element(&self, id: ElementId) -> Option<&InteractableElement> {
        self.element_map.get(&id).map(|idx| &self.elements[*idx])
    }

    /// Statistical summary of the scene.
    pub fn summary(&self) -> SceneSummary {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut total_volume = 0.0;
        let mut min_pos = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_pos = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
        for elem in &self.elements {
            let key = format!("{:?}", elem.interaction_type);
            *type_counts.entry(key).or_insert(0) += 1;
            total_volume += volume_size(&elem.activation_volume);
            min_pos.x = min_pos.x.min(elem.position[0]);
            min_pos.y = min_pos.y.min(elem.position[1]);
            min_pos.z = min_pos.z.min(elem.position[2]);
            max_pos.x = max_pos.x.max(elem.position[0]);
            max_pos.y = max_pos.y.max(elem.position[1]);
            max_pos.z = max_pos.z.max(elem.position[2]);
        }
        let extent = if self.elements.is_empty() {
            Vector3::zeros()
        } else {
            max_pos - min_pos
        };
        SceneSummary {
            element_count: self.elements.len(),
            type_counts,
            total_volume,
            scene_extent: extent,
        }
    }
}

/// Summary statistics for a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSummary {
    pub element_count: usize,
    pub type_counts: HashMap<String, usize>,
    pub total_volume: f64,
    pub scene_extent: Vector3,
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::geometry::Sphere;
    use xr_types::scene::DependencyType;

    fn make_scene() -> SceneModel {
        let mut scene = SceneModel::default();
        let mut btn = InteractableElement::new(
            "button",
            [1.0, 0.0, 0.0],
            InteractionType::Click,
        );
        btn.activation_volume = Volume::Sphere(Sphere {
            center: [1.0, 0.0, 0.0],
            radius: 0.1,
        });
        scene.add_element(btn);

        let mut slider = InteractableElement::new(
            "slider",
            [2.0, 0.0, 0.0],
            InteractionType::Slider,
        );
        slider.activation_volume = Volume::Box(BoundingBox::from_center_extents(
            [2.0, 0.0, 0.0],
            [0.3, 0.05, 0.05],
        ));
        scene.add_element(slider);

        let grab = InteractableElement::new(
            "handle",
            [0.0, 1.0, 0.0],
            InteractionType::Grab,
        );
        scene.add_element(grab);

        let dial = InteractableElement::new(
            "dial",
            [0.0, 0.0, 1.0],
            InteractionType::Dial,
        );
        scene.add_element(dial);

        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene
    }

    #[test]
    fn test_find_by_type() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let clicks = engine.find_by_type(InteractionType::Click);
        assert_eq!(clicks.len(), 1);
        assert_eq!(clicks[0].name, "button");
    }

    #[test]
    fn test_find_nearest() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let nearest = engine.find_nearest(&Point3::new(1.0, 0.0, 0.0), 2);
        assert!(!nearest.is_empty());
        assert_eq!(nearest[0].name, "button");
    }

    #[test]
    fn test_find_within_radius() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let results = engine.find_within_radius(&Point3::origin(), 1.5);
        assert!(results.len() >= 3);
    }

    #[test]
    fn test_find_in_box() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let bb = BoundingBox::from_center_extents(
            [1.5, 0.0, 0.0],
            [1.0, 0.5, 0.5],
        );
        let results = engine.find_in_box(&bb);
        assert!(results.len() >= 1);
    }

    #[test]
    fn test_frustum_query() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let frustum = Frustum::new(
            Point3::origin(),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            0.1,
            5.0,
        );
        let results = engine.frustum_query(&frustum);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filtered_query() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let filter = QueryFilter::new()
            .with_interaction_types(vec![InteractionType::Click, InteractionType::Slider]);
        let results = engine.filtered_query(&Point3::origin(), 10.0, &filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_group_by_type() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let groups = engine.group_by_type();
        assert!(groups.contains_key("Click"));
        assert!(groups.contains_key("Slider"));
    }

    #[test]
    fn test_density_map() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let density = engine.density_map(1.0);
        assert!(!density.is_empty());
    }

    #[test]
    fn test_summary() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let summary = engine.summary();
        assert_eq!(summary.element_count, 4);
        assert!(summary.total_volume > 0.0);
    }

    #[test]
    fn test_get_element() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let id = scene.elements[0].id;
        let elem = engine.get_element(id);
        assert!(elem.is_some());
        assert_eq!(elem.unwrap().name, "button");
    }

    #[test]
    fn test_find_reachable() {
        let scene = make_scene();
        let engine = SceneQueryEngine::from_scene(&scene);
        let reachable = engine.find_reachable(scene.elements[0].id);
        assert!(!reachable.is_empty());
    }
}
