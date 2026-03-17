//! Geometric separability for compositional verification.
//!
//! Builds spatial interference graphs, checks separability via convex hull
//! disjointness, computes tree decompositions using elimination ordering
//! heuristics, and finds spatial separators between interaction zones.

use std::collections::{HashMap, HashSet, VecDeque};

use choreo_types::geometry::AABB;
use choreo_types::spatial::{
    RegionId, SpatialPredicateId, ZoneId,
};

// ─── spatial interference graph ──────────────────────────────────────────────

/// Nodes are interaction zones, edges connect zones with shared predicates.
#[derive(Debug, Clone)]
pub struct SpatialInterferenceGraph {
    /// Zone → set of predicate IDs active in that zone.
    zone_predicates: HashMap<ZoneId, HashSet<SpatialPredicateId>>,
    /// Adjacency list: zone → set of interfering zones.
    adjacency: HashMap<ZoneId, HashSet<ZoneId>>,
    /// All zones.
    zones: Vec<ZoneId>,
    /// Zone → region data (AABB for separability checks).
    zone_bounds: HashMap<ZoneId, AABB>,
}

impl SpatialInterferenceGraph {
    pub fn new() -> Self {
        Self {
            zone_predicates: HashMap::new(),
            adjacency: HashMap::new(),
            zones: Vec::new(),
            zone_bounds: HashMap::new(),
        }
    }

    /// Add a zone with its associated predicates and bounding box.
    pub fn add_zone(
        &mut self,
        zone: ZoneId,
        predicates: HashSet<SpatialPredicateId>,
        bounds: AABB,
    ) {
        self.zone_predicates.insert(zone.clone(), predicates);
        self.zone_bounds.insert(zone.clone(), bounds);
        self.adjacency.entry(zone.clone()).or_default();
        self.zones.push(zone);
    }

    /// Compute interference edges: two zones interfere if they share a predicate.
    pub fn compute_edges(&mut self) {
        let zones: Vec<ZoneId> = self.zones.clone();
        for i in 0..zones.len() {
            for j in (i + 1)..zones.len() {
                let preds_i = self
                    .zone_predicates
                    .get(&zones[i])
                    .cloned()
                    .unwrap_or_default();
                let preds_j = self
                    .zone_predicates
                    .get(&zones[j])
                    .cloned()
                    .unwrap_or_default();

                if !preds_i.is_disjoint(&preds_j) {
                    self.adjacency
                        .entry(zones[i].clone())
                        .or_default()
                        .insert(zones[j].clone());
                    self.adjacency
                        .entry(zones[j].clone())
                        .or_default()
                        .insert(zones[i].clone());
                }
            }
        }
    }

    /// Get neighbours of a zone.
    pub fn neighbours(&self, zone: &ZoneId) -> Vec<&ZoneId> {
        self.adjacency
            .get(zone)
            .map(|s| s.iter().collect())
            .unwrap_or_default()
    }

    /// Number of zones.
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    /// Number of interference edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency
            .values()
            .map(|s| s.len())
            .sum::<usize>()
            / 2
    }

    /// Get all zones.
    pub fn zones(&self) -> &[ZoneId] {
        &self.zones
    }
}

impl Default for SpatialInterferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Build an interference graph from zones and their predicates.
pub fn compute_interference_graph(
    zone_predicates: &[(ZoneId, HashSet<SpatialPredicateId>, AABB)],
) -> SpatialInterferenceGraph {
    let mut graph = SpatialInterferenceGraph::new();
    for (zone, preds, bounds) in zone_predicates {
        graph.add_zone(zone.clone(), preds.clone(), *bounds);
    }
    graph.compute_edges();
    graph
}

// ─── separability checking ───────────────────────────────────────────────────

/// Check if two zones are separable (their convex hulls are disjoint).
pub fn check_separability(
    zone_a: &ZoneId,
    zone_b: &ZoneId,
    graph: &SpatialInterferenceGraph,
) -> bool {
    let bounds_a = graph.zone_bounds.get(zone_a);
    let bounds_b = graph.zone_bounds.get(zone_b);

    match (bounds_a, bounds_b) {
        (Some(a), Some(b)) => !a.intersects(b),
        _ => false,
    }
}

/// Compute the overlap bound (volume of AABB intersection) between two zones.
pub fn compute_overlap_bound(
    zone_a: &ZoneId,
    zone_b: &ZoneId,
    graph: &SpatialInterferenceGraph,
) -> f64 {
    let bounds_a = graph.zone_bounds.get(zone_a);
    let bounds_b = graph.zone_bounds.get(zone_b);

    match (bounds_a, bounds_b) {
        (Some(a), Some(b)) => {
            if !a.intersects(b) {
                return 0.0;
            }
            let inter = AABB::new(
                [
                    a.min[0].max(b.min[0]),
                    a.min[1].max(b.min[1]),
                    a.min[2].max(b.min[2]),
                ],
                [
                    a.max[0].min(b.max[0]),
                    a.max[1].min(b.max[1]),
                    a.max[2].min(b.max[2]),
                ],
            );
            inter.volume()
        }
        _ => 0.0,
    }
}

// ─── tree decomposition ──────────────────────────────────────────────────────

/// A tree decomposition of the interference graph.
#[derive(Debug, Clone)]
pub struct TreeDecomposition {
    /// Each bag is a set of zones.
    pub bags: Vec<TreeBag>,
    /// Edges between bags (indices into `bags`).
    pub edges: Vec<(usize, usize)>,
    /// Width of the decomposition (max bag size - 1).
    pub width: usize,
}

/// A bag in the tree decomposition.
#[derive(Debug, Clone)]
pub struct TreeBag {
    pub id: usize,
    pub zones: HashSet<ZoneId>,
}

impl TreeDecomposition {
    pub fn new() -> Self {
        Self {
            bags: Vec::new(),
            edges: Vec::new(),
            width: 0,
        }
    }

    /// Get the width (= max bag size - 1).
    pub fn treewidth(&self) -> usize {
        self.width
    }

    /// Number of bags.
    pub fn bag_count(&self) -> usize {
        self.bags.len()
    }
}

impl Default for TreeDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a tree decomposition using the minimum-degree elimination ordering heuristic.
pub fn compute_tree_decomposition(graph: &SpatialInterferenceGraph) -> TreeDecomposition {
    if graph.zones.is_empty() {
        return TreeDecomposition::new();
    }

    // Build adjacency as index-based graph.
    let zone_to_idx: HashMap<ZoneId, usize> = graph
        .zones
        .iter()
        .enumerate()
        .map(|(i, z)| (z.clone(), i))
        .collect();
    let n = graph.zones.len();

    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (z, nbrs) in &graph.adjacency {
        if let Some(&zi) = zone_to_idx.get(z) {
            for nbr in nbrs {
                if let Some(&ni) = zone_to_idx.get(nbr) {
                    adj[zi].insert(ni);
                    adj[ni].insert(zi);
                }
            }
        }
    }

    // Minimum-degree elimination ordering.
    let mut eliminated = vec![false; n];
    let mut elimination_order = Vec::new();
    let mut bags_map: HashMap<usize, HashSet<usize>> = HashMap::new();

    for _ in 0..n {
        // Find non-eliminated vertex with minimum degree.
        let mut min_deg = usize::MAX;
        let mut min_v = 0;
        for v in 0..n {
            if eliminated[v] {
                continue;
            }
            let deg = adj[v].iter().filter(|&&u| !eliminated[u]).count();
            if deg < min_deg {
                min_deg = deg;
                min_v = v;
            }
        }

        // The bag for this step = {min_v} ∪ neighbours of min_v.
        let mut bag: HashSet<usize> = adj[min_v]
            .iter()
            .copied()
            .filter(|&u| !eliminated[u])
            .collect();
        bag.insert(min_v);

        // Make neighbours into a clique (fill-in).
        let nbrs: Vec<usize> = bag.iter().copied().filter(|&v| v != min_v).collect();
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                adj[nbrs[i]].insert(nbrs[j]);
                adj[nbrs[j]].insert(nbrs[i]);
            }
        }

        bags_map.insert(elimination_order.len(), bag);
        elimination_order.push(min_v);
        eliminated[min_v] = true;
    }

    // Build tree decomposition from elimination ordering.
    let mut bags: Vec<TreeBag> = Vec::new();
    let mut edges = Vec::new();
    let mut vertex_last_bag: HashMap<usize, usize> = HashMap::new();

    for (step, &v) in elimination_order.iter().enumerate() {
        let bag_indices = &bags_map[&step];
        let zone_set: HashSet<ZoneId> = bag_indices
            .iter()
            .map(|&i| graph.zones[i].clone())
            .collect();

        let bag_id = bags.len();
        bags.push(TreeBag {
            id: bag_id,
            zones: zone_set,
        });

        // Connect to the last bag containing one of the vertices in this bag.
        for &u in bag_indices {
            if u != v {
                if let Some(&prev_bag) = vertex_last_bag.get(&u) {
                    if prev_bag != bag_id
                        && !edges.contains(&(prev_bag, bag_id))
                        && !edges.contains(&(bag_id, prev_bag))
                    {
                        edges.push((prev_bag, bag_id));
                    }
                }
            }
        }

        for &u in bag_indices {
            vertex_last_bag.insert(u, bag_id);
        }
    }

    let width = bags.iter().map(|b| b.zones.len()).max().unwrap_or(1) - 1;

    TreeDecomposition { bags, edges, width }
}

// ─── spatial separator ───────────────────────────────────────────────────────

/// A spatial separator: a set of regions that separates interaction zones.
#[derive(Debug, Clone)]
pub struct SpatialSeparator {
    /// Separator regions.
    pub regions: Vec<RegionId>,
    /// Zones on side A.
    pub side_a: Vec<ZoneId>,
    /// Zones on side B.
    pub side_b: Vec<ZoneId>,
}

/// Find a spatial separator between two sets of zones.
pub fn find_separator(
    graph: &SpatialInterferenceGraph,
    side_a: &[ZoneId],
    side_b: &[ZoneId],
) -> Option<SpatialSeparator> {
    // Find zones that have edges to both sides.
    let side_a_set: HashSet<_> = side_a.iter().collect();
    let side_b_set: HashSet<_> = side_b.iter().collect();

    let mut separator_zones = Vec::new();

    for zone in &graph.zones {
        if side_a_set.contains(zone) || side_b_set.contains(zone) {
            continue;
        }
        let nbrs = graph.neighbours(zone);
        let has_a_nbr = nbrs.iter().any(|n| side_a_set.contains(n));
        let has_b_nbr = nbrs.iter().any(|n| side_b_set.contains(n));
        if has_a_nbr && has_b_nbr {
            separator_zones.push(RegionId(zone.0.clone()));
        }
    }

    if separator_zones.is_empty() {
        None
    } else {
        Some(SpatialSeparator {
            regions: separator_zones,
            side_a: side_a.to_vec(),
            side_b: side_b.to_vec(),
        })
    }
}

// ─── component decomposition ─────────────────────────────────────────────────

/// A group of zones for compositional verification.
#[derive(Debug, Clone)]
pub struct ComponentGroup {
    pub zones: Vec<ZoneId>,
    pub shared_predicates: HashSet<SpatialPredicateId>,
}

/// Decompose zones into independent component groups for compositional verification.
pub fn decompose_for_composition(
    graph: &SpatialInterferenceGraph,
) -> Vec<ComponentGroup> {
    if graph.zones.is_empty() {
        return vec![];
    }

    // Find connected components.
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for zone in &graph.zones {
        if visited.contains(zone) {
            continue;
        }

        let mut component = Vec::new();
        let mut shared_preds = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(zone.clone());
        visited.insert(zone.clone());

        while let Some(current) = queue.pop_front() {
            component.push(current.clone());

            // Collect predicates shared with neighbours.
            if let Some(preds) = graph.zone_predicates.get(&current) {
                for nbr in graph.neighbours(&current) {
                    if let Some(nbr_preds) = graph.zone_predicates.get(nbr) {
                        let shared: HashSet<_> = preds.intersection(nbr_preds).cloned().collect();
                        shared_preds.extend(shared);
                    }
                }
            }

            for nbr in graph.neighbours(&current) {
                if visited.insert(nbr.clone()) {
                    queue.push_back(nbr.clone());
                }
            }
        }

        components.push(ComponentGroup {
            zones: component,
            shared_predicates: shared_preds,
        });
    }

    components
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zid(s: &str) -> ZoneId {
        ZoneId(s.to_string())
    }

    fn pid(s: &str) -> SpatialPredicateId {
        SpatialPredicateId(s.to_string())
    }

    fn make_bounds(x: f64, y: f64, z: f64, size: f64) -> AABB {
        AABB::new(
            [x - size, y - size, z - size],
            [x + size, y + size, z + size],
        )
    }

    #[test]
    fn test_interference_graph_no_shared_predicates() {
        let zone_data = vec![
            (zid("a"), [pid("p1")].into_iter().collect(), make_bounds(0.0, 0.0, 0.0, 1.0)),
            (zid("b"), [pid("p2")].into_iter().collect(), make_bounds(5.0, 0.0, 0.0, 1.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        assert_eq!(graph.zone_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_interference_graph_shared_predicates() {
        let zone_data = vec![
            (
                zid("a"),
                [pid("p1"), pid("p_shared")].into_iter().collect(),
                make_bounds(0.0, 0.0, 0.0, 1.0),
            ),
            (
                zid("b"),
                [pid("p2"), pid("p_shared")].into_iter().collect(),
                make_bounds(5.0, 0.0, 0.0, 1.0),
            ),
        ];
        let graph = compute_interference_graph(&zone_data);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_separability_disjoint() {
        let zone_data = vec![
            (zid("a"), HashSet::new(), make_bounds(0.0, 0.0, 0.0, 1.0)),
            (zid("b"), HashSet::new(), make_bounds(10.0, 0.0, 0.0, 1.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        assert!(check_separability(&zid("a"), &zid("b"), &graph));
    }

    #[test]
    fn test_separability_overlapping() {
        let zone_data = vec![
            (zid("a"), HashSet::new(), make_bounds(0.0, 0.0, 0.0, 2.0)),
            (zid("b"), HashSet::new(), make_bounds(1.0, 0.0, 0.0, 2.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        assert!(!check_separability(&zid("a"), &zid("b"), &graph));
    }

    #[test]
    fn test_overlap_bound() {
        let zone_data = vec![
            (zid("a"), HashSet::new(), AABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])),
            (zid("b"), HashSet::new(), AABB::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0])),
        ];
        let graph = compute_interference_graph(&zone_data);
        let overlap = compute_overlap_bound(&zid("a"), &zid("b"), &graph);
        assert!((overlap - 1.0).abs() < 0.01); // 1x1x1 = 1.0
    }

    #[test]
    fn test_tree_decomposition_empty() {
        let graph = SpatialInterferenceGraph::new();
        let td = compute_tree_decomposition(&graph);
        assert_eq!(td.bag_count(), 0);
    }

    #[test]
    fn test_tree_decomposition_single_zone() {
        let zone_data = vec![(zid("a"), HashSet::new(), make_bounds(0.0, 0.0, 0.0, 1.0))];
        let graph = compute_interference_graph(&zone_data);
        let td = compute_tree_decomposition(&graph);
        assert_eq!(td.bag_count(), 1);
        assert_eq!(td.treewidth(), 0);
    }

    #[test]
    fn test_tree_decomposition_path() {
        // Linear chain: a—b—c
        let zone_data = vec![
            (
                zid("a"),
                [pid("p_ab")].into_iter().collect(),
                make_bounds(0.0, 0.0, 0.0, 1.0),
            ),
            (
                zid("b"),
                [pid("p_ab"), pid("p_bc")].into_iter().collect(),
                make_bounds(1.0, 0.0, 0.0, 1.0),
            ),
            (
                zid("c"),
                [pid("p_bc")].into_iter().collect(),
                make_bounds(2.0, 0.0, 0.0, 1.0),
            ),
        ];
        let graph = compute_interference_graph(&zone_data);
        let td = compute_tree_decomposition(&graph);
        assert!(td.bag_count() >= 1);
        // Treewidth of a path is 1.
        assert!(td.treewidth() <= 2);
    }

    #[test]
    fn test_decompose_for_composition() {
        // Two disconnected components.
        let zone_data = vec![
            (zid("a"), [pid("p1")].into_iter().collect(), make_bounds(0.0, 0.0, 0.0, 1.0)),
            (zid("b"), [pid("p1")].into_iter().collect(), make_bounds(1.0, 0.0, 0.0, 1.0)),
            (zid("c"), [pid("p2")].into_iter().collect(), make_bounds(10.0, 0.0, 0.0, 1.0)),
            (zid("d"), [pid("p2")].into_iter().collect(), make_bounds(11.0, 0.0, 0.0, 1.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        let groups = decompose_for_composition(&graph);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_decompose_single_component() {
        let shared: HashSet<_> = [pid("p1")].into_iter().collect();
        let zone_data = vec![
            (zid("a"), shared.clone(), make_bounds(0.0, 0.0, 0.0, 1.0)),
            (zid("b"), shared.clone(), make_bounds(1.0, 0.0, 0.0, 1.0)),
            (zid("c"), shared.clone(), make_bounds(2.0, 0.0, 0.0, 1.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        let groups = decompose_for_composition(&graph);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].zones.len(), 3);
    }

    #[test]
    fn test_find_separator() {
        let zone_data = vec![
            (zid("a"), [pid("p1")].into_iter().collect(), make_bounds(0.0, 0.0, 0.0, 1.0)),
            (zid("sep"), [pid("p1"), pid("p2")].into_iter().collect(), make_bounds(3.0, 0.0, 0.0, 1.0)),
            (zid("b"), [pid("p2")].into_iter().collect(), make_bounds(6.0, 0.0, 0.0, 1.0)),
        ];
        let graph = compute_interference_graph(&zone_data);
        let sep = find_separator(&graph, &[zid("a")], &[zid("b")]);
        assert!(sep.is_some());
        let s = sep.unwrap();
        assert!(!s.regions.is_empty());
    }
}
