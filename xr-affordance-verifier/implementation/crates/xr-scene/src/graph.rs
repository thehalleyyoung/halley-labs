//! Scene graph construction and analysis using petgraph.
//!
//! Provides `SceneGraph` for representing scene elements and their dependencies
//! as a directed graph, with methods for topological traversal, cycle detection,
//! connected component analysis, and interaction cluster decomposition.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use xr_types::geometry::BoundingBox;
use xr_types::scene::{DependencyEdge, DependencyType, InteractableElement, InteractionType, SceneModel};

/// A node in the scene graph holding element data and spatial information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    pub element_index: usize,
    pub id: Uuid,
    pub name: String,
    pub position: [f64; 3],
    pub bounds: BoundingBox,
    pub interaction_type: InteractionType,
    pub tags: Vec<String>,
}

impl SceneNode {
    pub fn from_element(element: &InteractableElement, index: usize) -> Self {
        Self {
            element_index: index,
            id: element.id,
            name: element.name.clone(),
            position: element.position,
            bounds: element.activation_volume.bounding_box(),
            interaction_type: element.interaction_type,
            tags: element.tags.clone(),
        }
    }
}

/// An edge in the scene graph representing a dependency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SceneEdge {
    pub dependency_type: DependencyType,
    pub weight: f64,
}

impl SceneEdge {
    pub fn new(dep_type: DependencyType) -> Self {
        let weight = match dep_type {
            DependencyType::Sequential => 1.0,
            DependencyType::Visibility => 0.8,
            DependencyType::Enable => 0.9,
            DependencyType::Concurrent => 1.5,
            DependencyType::Unlock => 1.2,
        };
        Self {
            dependency_type: dep_type,
            weight,
        }
    }
}

/// A cluster of related interactions that can be analyzed together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionCluster {
    pub id: usize,
    pub node_indices: Vec<usize>,
    pub element_names: Vec<String>,
    pub interaction_types: Vec<InteractionType>,
    pub bounds: BoundingBox,
    pub max_depth: usize,
    pub edge_count: usize,
}

impl InteractionCluster {
    pub fn element_count(&self) -> usize {
        self.node_indices.len()
    }

    pub fn contains_type(&self, itype: InteractionType) -> bool {
        self.interaction_types.contains(&itype)
    }

    /// Estimated complexity for verification.
    pub fn complexity_estimate(&self) -> f64 {
        let n = self.node_indices.len() as f64;
        let e = self.edge_count as f64;
        let d = self.max_depth as f64;
        n * (1.0 + e / n.max(1.0)) * (1.0 + d * 0.5)
    }
}

/// A directed scene graph built from a SceneModel.
pub struct SceneGraph {
    graph: DiGraph<SceneNode, SceneEdge>,
    element_to_node: HashMap<usize, NodeIndex>,
    id_to_node: HashMap<Uuid, NodeIndex>,
}

impl SceneGraph {
    /// Build a scene graph from a SceneModel.
    pub fn from_scene(scene: &SceneModel) -> Self {
        let mut graph = DiGraph::new();
        let mut element_to_node = HashMap::new();
        let mut id_to_node = HashMap::new();

        for (i, elem) in scene.elements.iter().enumerate() {
            let node = SceneNode::from_element(elem, i);
            let id = elem.id;
            let idx = graph.add_node(node);
            element_to_node.insert(i, idx);
            id_to_node.insert(id, idx);
        }

        for dep in &scene.dependencies {
            if let (Some(&src), Some(&tgt)) = (
                element_to_node.get(&dep.source_index),
                element_to_node.get(&dep.target_index),
            ) {
                graph.add_edge(src, tgt, SceneEdge::new(dep.dependency_type));
            }
        }

        Self {
            graph,
            element_to_node,
            id_to_node,
        }
    }

    /// Create an empty scene graph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            element_to_node: HashMap::new(),
            id_to_node: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: SceneNode) -> NodeIndex {
        let elem_idx = node.element_index;
        let id = node.id;
        let ni = self.graph.add_node(node);
        self.element_to_node.insert(elem_idx, ni);
        self.id_to_node.insert(id, ni);
        ni
    }

    /// Add an edge between two element indices.
    pub fn add_edge(&mut self, source: usize, target: usize, dep_type: DependencyType) -> bool {
        if let (Some(&src), Some(&tgt)) = (
            self.element_to_node.get(&source),
            self.element_to_node.get(&target),
        ) {
            self.graph.add_edge(src, tgt, SceneEdge::new(dep_type));
            true
        } else {
            false
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get node data by element index.
    pub fn node_by_element(&self, element_index: usize) -> Option<&SceneNode> {
        self.element_to_node
            .get(&element_index)
            .and_then(|ni| self.graph.node_weight(*ni))
    }

    /// Get node data by UUID.
    pub fn node_by_id(&self, id: Uuid) -> Option<&SceneNode> {
        self.id_to_node
            .get(&id)
            .and_then(|ni| self.graph.node_weight(*ni))
    }

    /// Topological traversal of the graph (Kahn's algorithm).
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
        for ni in self.graph.node_indices() {
            in_degree.insert(ni, 0);
        }
        for edge in self.graph.edge_references() {
            *in_degree.entry(edge.target()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<NodeIndex> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&ni, _)| ni)
            .collect();

        let mut order = Vec::with_capacity(self.graph.node_count());
        while let Some(ni) = queue.pop_front() {
            if let Some(node) = self.graph.node_weight(ni) {
                order.push(node.element_index);
            }
            for neighbor in self.graph.neighbors_directed(ni, Direction::Outgoing) {
                if let Some(deg) = in_degree.get_mut(&neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if order.len() == self.graph.node_count() {
            Some(order)
        } else {
            None
        }
    }

    /// Check if the graph is a DAG.
    pub fn is_dag(&self) -> bool {
        self.topological_order().is_some()
    }

    /// Detect cycles using DFS. Returns one cycle if found.
    pub fn find_cycle(&self) -> Option<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        for start in self.graph.node_indices() {
            if visited.contains(&start) {
                continue;
            }
            let mut stack = vec![(start, false)];
            while let Some((ni, backtrack)) = stack.pop() {
                if backtrack {
                    on_stack.remove(&ni);
                    continue;
                }
                if on_stack.contains(&ni) {
                    // Found a cycle — reconstruct it
                    let mut cycle = vec![];
                    let cycle_node = ni;
                    if let Some(node) = self.graph.node_weight(cycle_node) {
                        cycle.push(node.element_index);
                    }
                    let mut current = *parent.get(&ni).unwrap_or(&ni);
                    while current != cycle_node {
                        if let Some(node) = self.graph.node_weight(current) {
                            cycle.push(node.element_index);
                        }
                        current = *parent.get(&current).unwrap_or(&cycle_node);
                    }
                    cycle.reverse();
                    return Some(cycle);
                }
                if visited.contains(&ni) {
                    continue;
                }
                visited.insert(ni);
                on_stack.insert(ni);
                stack.push((ni, true));

                for neighbor in self.graph.neighbors_directed(ni, Direction::Outgoing) {
                    if !visited.contains(&neighbor) || on_stack.contains(&neighbor) {
                        parent.insert(neighbor, ni);
                        stack.push((neighbor, false));
                    }
                }
            }
        }
        None
    }

    /// Find connected components (treating edges as undirected).
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for start in self.graph.node_indices() {
            if visited.contains(&start) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);

            while let Some(ni) = queue.pop_front() {
                if let Some(node) = self.graph.node_weight(ni) {
                    component.push(node.element_index);
                }
                for neighbor in self
                    .graph
                    .neighbors_directed(ni, Direction::Outgoing)
                    .chain(self.graph.neighbors_directed(ni, Direction::Incoming))
                {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            component.sort();
            components.push(component);
        }
        components
    }

    /// Extract a subgraph containing only the given element indices.
    pub fn subgraph(&self, element_indices: &[usize]) -> SceneGraph {
        let keep: HashSet<usize> = element_indices.iter().copied().collect();
        let mut new_graph = DiGraph::new();
        let mut new_elem_to_node = HashMap::new();
        let mut new_id_to_node = HashMap::new();

        for &elem_idx in element_indices {
            if let Some(&ni) = self.element_to_node.get(&elem_idx) {
                if let Some(node) = self.graph.node_weight(ni) {
                    let new_ni = new_graph.add_node(node.clone());
                    new_elem_to_node.insert(elem_idx, new_ni);
                    new_id_to_node.insert(node.id, new_ni);
                }
            }
        }

        for edge in self.graph.edge_references() {
            let src_node = self.graph.node_weight(edge.source());
            let tgt_node = self.graph.node_weight(edge.target());
            if let (Some(src), Some(tgt)) = (src_node, tgt_node) {
                if keep.contains(&src.element_index) && keep.contains(&tgt.element_index) {
                    if let (Some(&new_src), Some(&new_tgt)) = (
                        new_elem_to_node.get(&src.element_index),
                        new_elem_to_node.get(&tgt.element_index),
                    ) {
                        new_graph.add_edge(new_src, new_tgt, *edge.weight());
                    }
                }
            }
        }

        SceneGraph {
            graph: new_graph,
            element_to_node: new_elem_to_node,
            id_to_node: new_id_to_node,
        }
    }

    /// Get predecessor element indices for a given element.
    pub fn predecessors(&self, element_index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(&ni) = self.element_to_node.get(&element_index) {
            for neighbor in self.graph.neighbors_directed(ni, Direction::Incoming) {
                if let Some(node) = self.graph.node_weight(neighbor) {
                    result.push(node.element_index);
                }
            }
        }
        result
    }

    /// Get successor element indices for a given element.
    pub fn successors(&self, element_index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        if let Some(&ni) = self.element_to_node.get(&element_index) {
            for neighbor in self.graph.neighbors_directed(ni, Direction::Outgoing) {
                if let Some(node) = self.graph.node_weight(neighbor) {
                    result.push(node.element_index);
                }
            }
        }
        result
    }

    /// Find all root nodes (no incoming edges).
    pub fn roots(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for ni in self.graph.node_indices() {
            if self
                .graph
                .neighbors_directed(ni, Direction::Incoming)
                .next()
                .is_none()
            {
                if let Some(node) = self.graph.node_weight(ni) {
                    result.push(node.element_index);
                }
            }
        }
        result
    }

    /// Find all leaf nodes (no outgoing edges).
    pub fn leaves(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for ni in self.graph.node_indices() {
            if self
                .graph
                .neighbors_directed(ni, Direction::Outgoing)
                .next()
                .is_none()
            {
                if let Some(node) = self.graph.node_weight(ni) {
                    result.push(node.element_index);
                }
            }
        }
        result
    }

    /// Compute the depth (longest path from any root) for each element.
    pub fn element_depths(&self) -> HashMap<usize, usize> {
        let mut depths = HashMap::new();
        if let Some(topo) = self.topological_order() {
            for &elem in &topo {
                depths.insert(elem, 0);
            }
            for &elem in &topo {
                let d = depths[&elem];
                for succ in self.successors(elem) {
                    let entry = depths.entry(succ).or_insert(0);
                    *entry = (*entry).max(d + 1);
                }
            }
        }
        depths
    }

    /// Maximum depth across all elements.
    pub fn max_depth(&self) -> usize {
        self.element_depths().values().copied().max().unwrap_or(0)
    }

    /// Detect interaction clusters: connected components with metadata.
    pub fn interaction_clusters(&self) -> Vec<InteractionCluster> {
        let components = self.connected_components();
        let depths = self.element_depths();

        components
            .into_iter()
            .enumerate()
            .map(|(id, indices)| {
                let index_set: HashSet<usize> = indices.iter().copied().collect();
                let mut names = Vec::new();
                let mut itypes = Vec::new();
                let mut bounds: Option<BoundingBox> = None;
                let mut max_depth = 0usize;

                for &idx in &indices {
                    if let Some(node) = self.node_by_element(idx) {
                        names.push(node.name.clone());
                        if !itypes.contains(&node.interaction_type) {
                            itypes.push(node.interaction_type);
                        }
                        bounds = Some(match bounds {
                            Some(b) => b.union(&node.bounds),
                            None => node.bounds,
                        });
                    }
                    if let Some(&d) = depths.get(&idx) {
                        max_depth = max_depth.max(d);
                    }
                }

                // Count internal edges
                let mut edge_count = 0;
                for edge in self.graph.edge_references() {
                    let src = self.graph.node_weight(edge.source());
                    let tgt = self.graph.node_weight(edge.target());
                    if let (Some(s), Some(t)) = (src, tgt) {
                        if index_set.contains(&s.element_index)
                            && index_set.contains(&t.element_index)
                        {
                            edge_count += 1;
                        }
                    }
                }

                InteractionCluster {
                    id,
                    node_indices: indices,
                    element_names: names,
                    interaction_types: itypes,
                    bounds: bounds.unwrap_or_default(),
                    max_depth,
                    edge_count,
                }
            })
            .collect()
    }

    /// Estimate treewidth using a greedy min-degree elimination heuristic.
    pub fn estimate_treewidth(&self) -> usize {
        if self.graph.node_count() == 0 {
            return 0;
        }

        // Build undirected adjacency
        let n = self.graph.node_count();
        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let mut idx_map: HashMap<NodeIndex, usize> = HashMap::new();
        for (i, &ni) in node_indices.iter().enumerate() {
            idx_map.insert(ni, i);
        }

        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for edge in self.graph.edge_references() {
            if let (Some(&si), Some(&ti)) = (idx_map.get(&edge.source()), idx_map.get(&edge.target())) {
                adj[si].insert(ti);
                adj[ti].insert(si);
            }
        }

        let mut eliminated = vec![false; n];
        let mut treewidth = 0usize;

        for _ in 0..n {
            // Find the non-eliminated node with minimum degree
            let mut min_deg = usize::MAX;
            let mut min_node = 0;
            for i in 0..n {
                if eliminated[i] {
                    continue;
                }
                let deg = adj[i].iter().filter(|&&j| !eliminated[j]).count();
                if deg < min_deg {
                    min_deg = deg;
                    min_node = i;
                }
            }

            treewidth = treewidth.max(min_deg);

            // Connect all neighbors of min_node to each other (elimination step)
            let neighbors: Vec<usize> = adj[min_node]
                .iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    adj[neighbors[i]].insert(neighbors[j]);
                    adj[neighbors[j]].insert(neighbors[i]);
                }
            }
            eliminated[min_node] = true;
        }

        treewidth
    }

    /// Find all paths between two elements (up to a maximum count).
    pub fn all_paths(&self, from: usize, to: usize, max_paths: usize) -> Vec<Vec<usize>> {
        let mut results = Vec::new();
        let start = match self.element_to_node.get(&from) {
            Some(&ni) => ni,
            None => return results,
        };
        let end = match self.element_to_node.get(&to) {
            Some(&ni) => ni,
            None => return results,
        };

        let mut stack: Vec<(NodeIndex, Vec<NodeIndex>)> = vec![(start, vec![start])];
        let mut visited_paths = 0usize;

        while let Some((current, path)) = stack.pop() {
            if visited_paths >= max_paths {
                break;
            }
            if current == end {
                let elem_path: Vec<usize> = path
                    .iter()
                    .filter_map(|&ni| self.graph.node_weight(ni).map(|n| n.element_index))
                    .collect();
                results.push(elem_path);
                visited_paths += 1;
                continue;
            }
            for neighbor in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if !path.contains(&neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    stack.push((neighbor, new_path));
                }
            }
        }
        results
    }

    /// Get the longest path in the graph (critical path).
    pub fn critical_path(&self) -> Vec<usize> {
        let depths = self.element_depths();
        let max_d = depths.values().copied().max().unwrap_or(0);

        // Find a leaf with max depth
        let target = match depths.iter().find(|(_, &d)| d == max_d) {
            Some((&elem, _)) => elem,
            None => return vec![],
        };

        // Backtrack from target to root
        let mut path = vec![target];
        let mut current = target;
        loop {
            let preds = self.predecessors(current);
            if preds.is_empty() {
                break;
            }
            // Pick the predecessor with max depth
            let best = preds
                .into_iter()
                .max_by_key(|p| depths.get(p).copied().unwrap_or(0))
                .unwrap();
            path.push(best);
            current = best;
        }
        path.reverse();
        path
    }

    /// Compute edges by dependency type.
    pub fn edges_by_type(&self) -> HashMap<DependencyType, usize> {
        let mut counts = HashMap::new();
        for edge in self.graph.edge_references() {
            *counts.entry(edge.weight().dependency_type).or_insert(0) += 1;
        }
        counts
    }

    /// Get the strongly connected components.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let sccs = petgraph::algo::kosaraju_scc(&self.graph);
        sccs.into_iter()
            .map(|scc| {
                scc.into_iter()
                    .filter_map(|ni| self.graph.node_weight(ni).map(|n| n.element_index))
                    .collect()
            })
            .collect()
    }

    /// Convert back to dependency edges.
    pub fn to_dependency_edges(&self) -> Vec<DependencyEdge> {
        self.graph
            .edge_references()
            .filter_map(|edge| {
                let src = self.graph.node_weight(edge.source())?;
                let tgt = self.graph.node_weight(edge.target())?;
                Some(DependencyEdge {
                    source_index: src.element_index,
                    target_index: tgt.element_index,
                    dependency_type: edge.weight().dependency_type,
                })
            })
            .collect()
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::scene::InteractableElement;

    fn make_scene() -> SceneModel {
        let mut scene = SceneModel::new("TestGraph");
        scene.add_element(InteractableElement::new("A", [0.0, 0.0, 0.0], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Grab));
        scene.add_element(InteractableElement::new("C", [2.0, 0.0, 0.0], InteractionType::Drag));
        scene.add_element(InteractableElement::new("D", [3.0, 0.0, 0.0], InteractionType::Click));
        scene.add_dependency(0, 1, DependencyType::Sequential);
        scene.add_dependency(0, 2, DependencyType::Enable);
        scene.add_dependency(1, 3, DependencyType::Sequential);
        scene.add_dependency(2, 3, DependencyType::Sequential);
        scene
    }

    #[test]
    fn test_build_graph() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
    }

    #[test]
    fn test_topological_order() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        // A must come before B and C; B and C must come before D
        let pos_a = order.iter().position(|&x| x == 0).unwrap();
        let pos_b = order.iter().position(|&x| x == 1).unwrap();
        let pos_c = order.iter().position(|&x| x == 2).unwrap();
        let pos_d = order.iter().position(|&x| x == 3).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_is_dag() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        assert!(graph.is_dag());
    }

    #[test]
    fn test_connected_components() {
        let mut scene = SceneModel::new("Test");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Click));
        scene.add_element(InteractableElement::new("C", [2.0, 0.0, 0.0], InteractionType::Click));
        scene.add_dependency(0, 1, DependencyType::Sequential);
        // C is isolated

        let graph = SceneGraph::from_scene(&scene);
        let components = graph.connected_components();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_subgraph() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let sub = graph.subgraph(&[0, 1]);
        assert_eq!(sub.node_count(), 2);
        assert_eq!(sub.edge_count(), 1);
    }

    #[test]
    fn test_roots_and_leaves() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        assert_eq!(graph.roots(), vec![0]);
        assert_eq!(graph.leaves(), vec![3]);
    }

    #[test]
    fn test_depths() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let depths = graph.element_depths();
        assert_eq!(depths[&0], 0);
        assert_eq!(depths[&1], 1);
        assert_eq!(depths[&2], 1);
        assert_eq!(depths[&3], 2);
        assert_eq!(graph.max_depth(), 2);
    }

    #[test]
    fn test_interaction_clusters() {
        let mut scene = SceneModel::new("Test");
        scene.add_element(InteractableElement::new("A", [0.0; 3], InteractionType::Click));
        scene.add_element(InteractableElement::new("B", [1.0, 0.0, 0.0], InteractionType::Grab));
        scene.add_element(InteractableElement::new("C", [5.0, 0.0, 0.0], InteractionType::Drag));
        scene.add_dependency(0, 1, DependencyType::Sequential);

        let graph = SceneGraph::from_scene(&scene);
        let clusters = graph.interaction_clusters();
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_treewidth_estimate() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let tw = graph.estimate_treewidth();
        // Diamond graph A→B,C→D has treewidth 2
        assert!(tw <= 3);
    }

    #[test]
    fn test_all_paths() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let paths = graph.all_paths(0, 3, 10);
        // Two paths: A→B→D and A→C→D
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_critical_path() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let cp = graph.critical_path();
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0], 0);
        assert_eq!(*cp.last().unwrap(), 3);
    }

    #[test]
    fn test_edges_by_type() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let counts = graph.edges_by_type();
        assert_eq!(counts[&DependencyType::Sequential], 3);
        assert_eq!(counts[&DependencyType::Enable], 1);
    }

    #[test]
    fn test_scc() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let sccs = graph.strongly_connected_components();
        // DAG → each SCC is a single node
        assert_eq!(sccs.len(), 4);
    }

    #[test]
    fn test_predecessors_successors() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let preds = graph.predecessors(3);
        assert!(preds.contains(&1));
        assert!(preds.contains(&2));
        let succs = graph.successors(0);
        assert!(succs.contains(&1));
        assert!(succs.contains(&2));
    }

    #[test]
    fn test_empty_graph() {
        let graph = SceneGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert!(graph.is_dag());
        assert_eq!(graph.max_depth(), 0);
        assert_eq!(graph.estimate_treewidth(), 0);
    }

    #[test]
    fn test_add_node_and_edge() {
        let mut graph = SceneGraph::new();
        let n0 = SceneNode {
            element_index: 0,
            id: Uuid::new_v4(),
            name: "A".to_string(),
            position: [0.0; 3],
            bounds: BoundingBox::default(),
            interaction_type: InteractionType::Click,
            tags: vec![],
        };
        let n1 = SceneNode {
            element_index: 1,
            id: Uuid::new_v4(),
            name: "B".to_string(),
            position: [1.0, 0.0, 0.0],
            bounds: BoundingBox::default(),
            interaction_type: InteractionType::Grab,
            tags: vec![],
        };
        graph.add_node(n0);
        graph.add_node(n1);
        assert!(graph.add_edge(0, 1, DependencyType::Sequential));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_to_dependency_edges() {
        let scene = make_scene();
        let graph = SceneGraph::from_scene(&scene);
        let edges = graph.to_dependency_edges();
        assert_eq!(edges.len(), 4);
    }
}
