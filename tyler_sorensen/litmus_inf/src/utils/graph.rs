//! Graph algorithms for execution graphs in LITMUS∞.
//!
//! Provides a generic directed graph, topological sort, strongly connected
//! components (Tarjan, Kosaraju), cycle detection and enumeration,
//! shortest paths (Dijkstra, BFS, Bellman-Ford, Floyd-Warshall),
//! graph isomorphism, coloring, dominator trees, traversals, and utilities.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque, BinaryHeap};
use std::fmt;
use std::cmp::Ordering;

// ═══════════════════════════════════════════════════════════════════════
// NodeId, EdgeId
// ═══════════════════════════════════════════════════════════════════════

/// Unique identifier for a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Unique identifier for a graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Graph<V, E> — generic directed graph
// ═══════════════════════════════════════════════════════════════════════

/// A node in the graph.
#[derive(Debug, Clone)]
struct NodeEntry<V> {
    data: V,
    outgoing: Vec<EdgeId>,
    incoming: Vec<EdgeId>,
}

/// An edge in the graph.
#[derive(Debug, Clone)]
struct EdgeEntry<E> {
    from: NodeId,
    to: NodeId,
    data: E,
}

/// A generic directed graph with node data V and edge data E.
#[derive(Debug, Clone)]
pub struct Graph<V, E> {
    nodes: Vec<Option<NodeEntry<V>>>,
    edges: Vec<Option<EdgeEntry<E>>>,
    node_count: usize,
    edge_count: usize,
}

impl<V: Clone, E: Clone> Graph<V, E> {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Add a node with the given data. Returns its ID.
    pub fn add_node(&mut self, data: V) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Some(NodeEntry {
            data,
            outgoing: Vec::new(),
            incoming: Vec::new(),
        }));
        self.node_count += 1;
        id
    }

    /// Add a directed edge from `from` to `to` with the given data.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, data: E) -> EdgeId {
        let id = EdgeId(self.edges.len());
        self.edges.push(Some(EdgeEntry { from, to, data }));
        if let Some(node) = &mut self.nodes[from.0] {
            node.outgoing.push(id);
        }
        if let Some(node) = &mut self.nodes[to.0] {
            node.incoming.push(id);
        }
        self.edge_count += 1;
        id
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Get all valid node IDs.
    pub fn nodes(&self) -> Vec<NodeId> {
        self.nodes.iter().enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|_| NodeId(i)))
            .collect()
    }

    /// Get all valid edge IDs.
    pub fn edges(&self) -> Vec<EdgeId> {
        self.edges.iter().enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|_| EdgeId(i)))
            .collect()
    }

    /// Get node data.
    pub fn node_data(&self, id: NodeId) -> Option<&V> {
        self.nodes.get(id.0).and_then(|n| n.as_ref().map(|n| &n.data))
    }

    /// Get mutable node data.
    pub fn node_data_mut(&mut self, id: NodeId) -> Option<&mut V> {
        self.nodes.get_mut(id.0).and_then(|n| n.as_mut().map(|n| &mut n.data))
    }

    /// Get edge data.
    pub fn edge_data(&self, id: EdgeId) -> Option<&E> {
        self.edges.get(id.0).and_then(|e| e.as_ref().map(|e| &e.data))
    }

    /// Get edge endpoints.
    pub fn edge_endpoints(&self, id: EdgeId) -> Option<(NodeId, NodeId)> {
        self.edges.get(id.0).and_then(|e| e.as_ref().map(|e| (e.from, e.to)))
    }

    /// Get successors of a node.
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes.get(id.0)
            .and_then(|n| n.as_ref())
            .map(|n| {
                n.outgoing.iter()
                    .filter_map(|&eid| self.edges[eid.0].as_ref().map(|e| e.to))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get predecessors of a node.
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes.get(id.0)
            .and_then(|n| n.as_ref())
            .map(|n| {
                n.incoming.iter()
                    .filter_map(|&eid| self.edges[eid.0].as_ref().map(|e| e.from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Alias for successors.
    pub fn neighbors(&self, id: NodeId) -> Vec<NodeId> {
        self.successors(id)
    }

    /// Out-degree.
    pub fn out_degree(&self, id: NodeId) -> usize {
        self.successors(id).len()
    }

    /// In-degree.
    pub fn in_degree(&self, id: NodeId) -> usize {
        self.predecessors(id).len()
    }

    /// Check if an edge exists.
    pub fn has_edge(&self, from: NodeId, to: NodeId) -> bool {
        self.successors(from).contains(&to)
    }

    /// Graph density.
    pub fn density(&self) -> f64 {
        let n = self.node_count as f64;
        if n <= 1.0 { return 0.0; }
        self.edge_count as f64 / (n * (n - 1.0))
    }

    /// Convert to adjacency matrix.
    pub fn to_adjacency_matrix(&self) -> Vec<Vec<bool>> {
        let nodes = self.nodes();
        let n = nodes.len();
        let id_map: HashMap<NodeId, usize> = nodes.iter().enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let mut matrix = vec![vec![false; n]; n];
        for &nid in &nodes {
            for succ in self.successors(nid) {
                if let (Some(&i), Some(&j)) = (id_map.get(&nid), id_map.get(&succ)) {
                    matrix[i][j] = true;
                }
            }
        }
        matrix
    }
}

/// Create a graph from an adjacency matrix.
pub fn from_adjacency_matrix(matrix: &[Vec<bool>]) -> Graph<(), ()> {
    let n = matrix.len();
    let mut g: Graph<(), ()> = Graph::new();
    let nodes: Vec<NodeId> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j] {
                g.add_edge(nodes[i], nodes[j], ());
            }
        }
    }
    g
}

// ═══════════════════════════════════════════════════════════════════════
// CycleError
// ═══════════════════════════════════════════════════════════════════════

/// Error indicating a cycle was found.
#[derive(Debug, Clone)]
pub struct CycleError {
    /// The cycle as a sequence of node IDs.
    pub cycle: Vec<NodeId>,
}

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle found: {:?}", self.cycle)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TopologicalSort
// ═══════════════════════════════════════════════════════════════════════

/// Topological sorting algorithms.
pub struct TopologicalSort;

impl TopologicalSort {
    /// Kahn's algorithm (BFS-based).
    pub fn kahn<V: Clone, E: Clone>(graph: &Graph<V, E>) -> Result<Vec<NodeId>, CycleError> {
        let nodes = graph.nodes();
        let mut in_degrees: HashMap<NodeId, usize> = HashMap::new();
        for &nid in &nodes {
            in_degrees.insert(nid, graph.in_degree(nid));
        }

        let mut queue: VecDeque<NodeId> = nodes.iter()
            .filter(|&&nid| in_degrees[&nid] == 0)
            .copied()
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node);
            for succ in graph.successors(node) {
                if let Some(deg) = in_degrees.get_mut(&succ) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(succ);
                    }
                }
            }
        }

        if result.len() != nodes.len() {
            Err(CycleError { cycle: Vec::new() })
        } else {
            Ok(result)
        }
    }

    /// DFS-based topological sort.
    pub fn dfs<V: Clone, E: Clone>(graph: &Graph<V, E>) -> Result<Vec<NodeId>, CycleError> {
        let nodes = graph.nodes();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();
        let mut result = Vec::new();

        for &node in &nodes {
            if !visited.contains(&node) {
                Self::dfs_visit(graph, node, &mut visited, &mut on_stack, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn dfs_visit<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        on_stack: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<(), CycleError> {
        visited.insert(node);
        on_stack.insert(node);

        for succ in graph.successors(node) {
            if on_stack.contains(&succ) {
                return Err(CycleError { cycle: vec![succ, node] });
            }
            if !visited.contains(&succ) {
                Self::dfs_visit(graph, succ, visited, on_stack, result)?;
            }
        }

        on_stack.remove(&node);
        result.push(node);
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// StronglyConnectedComponents — Tarjan's and Kosaraju's algorithms
// ═══════════════════════════════════════════════════════════════════════

/// Strongly connected components algorithms.
pub struct StronglyConnectedComponents;

/// Result of SCC computation.
#[derive(Debug, Clone)]
pub struct SccResult {
    /// Components (each is a list of node IDs).
    pub components: Vec<Vec<NodeId>>,
    /// Component assignment per node.
    pub component_of: HashMap<NodeId, usize>,
}

impl StronglyConnectedComponents {
    /// Tarjan's algorithm for SCCs.
    pub fn tarjan<V: Clone, E: Clone>(graph: &Graph<V, E>) -> SccResult {
        let nodes = graph.nodes();
        let mut index_counter: usize = 0;
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlinks: HashMap<NodeId, usize> = HashMap::new();
        let mut components = Vec::new();

        for &node in &nodes {
            if !indices.contains_key(&node) {
                Self::tarjan_visit(
                    graph, node, &mut index_counter, &mut stack, &mut on_stack,
                    &mut indices, &mut lowlinks, &mut components,
                );
            }
        }

        let mut component_of = HashMap::new();
        for (ci, comp) in components.iter().enumerate() {
            for &nid in comp {
                component_of.insert(nid, ci);
            }
        }

        SccResult { components, component_of }
    }

    fn tarjan_visit<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        index_counter: &mut usize,
        stack: &mut Vec<NodeId>,
        on_stack: &mut HashSet<NodeId>,
        indices: &mut HashMap<NodeId, usize>,
        lowlinks: &mut HashMap<NodeId, usize>,
        components: &mut Vec<Vec<NodeId>>,
    ) {
        let idx = *index_counter;
        *index_counter += 1;
        indices.insert(node, idx);
        lowlinks.insert(node, idx);
        stack.push(node);
        on_stack.insert(node);

        for succ in graph.successors(node) {
            if !indices.contains_key(&succ) {
                Self::tarjan_visit(graph, succ, index_counter, stack, on_stack, indices, lowlinks, components);
                let ll = lowlinks[&succ];
                if ll < lowlinks[&node] {
                    lowlinks.insert(node, ll);
                }
            } else if on_stack.contains(&succ) {
                let si = indices[&succ];
                if si < lowlinks[&node] {
                    lowlinks.insert(node, si);
                }
            }
        }

        if lowlinks[&node] == indices[&node] {
            let mut component = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(&w);
                component.push(w);
                if w == node { break; }
            }
            components.push(component);
        }
    }

    /// Kosaraju's algorithm.
    pub fn kosaraju<V: Clone, E: Clone>(graph: &Graph<V, E>) -> SccResult {
        let nodes = graph.nodes();

        // First pass: compute finish order
        let mut visited = HashSet::new();
        let mut finish_order = Vec::new();
        for &node in &nodes {
            if !visited.contains(&node) {
                Self::kosaraju_dfs1(graph, node, &mut visited, &mut finish_order);
            }
        }

        // Build reverse graph successors
        let mut rev_adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &nid in &nodes {
            rev_adj.insert(nid, Vec::new());
        }
        for &nid in &nodes {
            for succ in graph.successors(nid) {
                rev_adj.entry(succ).or_default().push(nid);
            }
        }

        // Second pass: find SCCs in reverse finish order
        visited.clear();
        let mut components = Vec::new();
        for &node in finish_order.iter().rev() {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                Self::kosaraju_dfs2(&rev_adj, node, &mut visited, &mut component);
                components.push(component);
            }
        }

        let mut component_of = HashMap::new();
        for (ci, comp) in components.iter().enumerate() {
            for &nid in comp {
                component_of.insert(nid, ci);
            }
        }

        SccResult { components, component_of }
    }

    fn kosaraju_dfs1<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        finish_order: &mut Vec<NodeId>,
    ) {
        visited.insert(node);
        for succ in graph.successors(node) {
            if !visited.contains(&succ) {
                Self::kosaraju_dfs1(graph, succ, visited, finish_order);
            }
        }
        finish_order.push(node);
    }

    fn kosaraju_dfs2(
        rev_adj: &HashMap<NodeId, Vec<NodeId>>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        component: &mut Vec<NodeId>,
    ) {
        visited.insert(node);
        component.push(node);
        if let Some(preds) = rev_adj.get(&node) {
            for &pred in preds {
                if !visited.contains(&pred) {
                    Self::kosaraju_dfs2(rev_adj, pred, visited, component);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CycleDetection
// ═══════════════════════════════════════════════════════════════════════

/// Cycle detection and enumeration.
pub struct CycleDetection;

impl CycleDetection {
    /// Check if the graph has any cycle.
    pub fn has_cycle<V: Clone, E: Clone>(graph: &Graph<V, E>) -> bool {
        TopologicalSort::kahn(graph).is_err()
    }

    /// Find a cycle if one exists.
    pub fn find_cycle<V: Clone, E: Clone>(graph: &Graph<V, E>) -> Option<Vec<NodeId>> {
        let nodes = graph.nodes();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();

        for &node in &nodes {
            if !visited.contains(&node) {
                if let Some(cycle) = Self::find_cycle_dfs(graph, node, &mut visited, &mut on_stack, &mut parent) {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn find_cycle_dfs<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        on_stack: &mut HashSet<NodeId>,
        parent: &mut HashMap<NodeId, NodeId>,
    ) -> Option<Vec<NodeId>> {
        visited.insert(node);
        on_stack.insert(node);

        for succ in graph.successors(node) {
            if on_stack.contains(&succ) {
                // Found cycle, reconstruct it
                let mut cycle = vec![succ];
                let mut current = node;
                while current != succ {
                    cycle.push(current);
                    current = match parent.get(&current) {
                        Some(&p) => p,
                        None => break,
                    };
                }
                cycle.reverse();
                return Some(cycle);
            }
            if !visited.contains(&succ) {
                parent.insert(succ, node);
                if let Some(cycle) = Self::find_cycle_dfs(graph, succ, visited, on_stack, parent) {
                    return Some(cycle);
                }
            }
        }

        on_stack.remove(&node);
        None
    }

    /// Enumerate all simple cycles up to a maximum length (Johnson's algorithm simplified).
    pub fn enumerate_cycles<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        max_length: usize,
    ) -> Vec<Vec<NodeId>> {
        let mut cycles = Vec::new();
        let nodes = graph.nodes();

        for &start in &nodes {
            let mut path = vec![start];
            let mut visited = HashSet::new();
            visited.insert(start);
            Self::enumerate_dfs(graph, start, start, &mut path, &mut visited, max_length, &mut cycles);
        }

        // Deduplicate (normalize cycles to start at smallest node)
        let mut unique: HashSet<Vec<usize>> = HashSet::new();
        let mut result = Vec::new();
        for cycle in &cycles {
            let mut normalized: Vec<usize> = cycle.iter().map(|n| n.0).collect();
            let min_pos = normalized.iter().enumerate()
                .min_by_key(|(_, v)| *v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            normalized.rotate_left(min_pos);
            if unique.insert(normalized) {
                result.push(cycle.clone());
            }
        }
        result
    }

    fn enumerate_dfs<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        start: NodeId,
        current: NodeId,
        path: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        max_length: usize,
        cycles: &mut Vec<Vec<NodeId>>,
    ) {
        if path.len() > max_length { return; }
        if cycles.len() > 1000 { return; } // Safety limit

        for succ in graph.successors(current) {
            if succ == start && path.len() >= 2 {
                cycles.push(path.clone());
            } else if !visited.contains(&succ) && succ.0 >= start.0 {
                visited.insert(succ);
                path.push(succ);
                Self::enumerate_dfs(graph, start, succ, path, visited, max_length, cycles);
                path.pop();
                visited.remove(&succ);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ShortestPath
// ═══════════════════════════════════════════════════════════════════════

/// Shortest path algorithms.
pub struct ShortestPath;

/// Entry for Dijkstra's priority queue.
#[derive(Debug, Clone)]
struct DijkstraEntry {
    node: NodeId,
    dist: f64,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}
impl Eq for DijkstraEntry {}
impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal) // Min-heap
    }
}

impl ShortestPath {
    /// BFS shortest path (unweighted).
    pub fn bfs<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        source: NodeId,
        target: NodeId,
    ) -> Option<Vec<NodeId>> {
        let mut visited = HashSet::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut queue = VecDeque::new();

        visited.insert(source);
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            if node == target {
                let mut path = vec![target];
                let mut current = target;
                while current != source {
                    current = parent[&current];
                    path.push(current);
                }
                path.reverse();
                return Some(path);
            }

            for succ in graph.successors(node) {
                if !visited.contains(&succ) {
                    visited.insert(succ);
                    parent.insert(succ, node);
                    queue.push_back(succ);
                }
            }
        }

        None
    }

    /// Floyd-Warshall all-pairs shortest paths.
    pub fn floyd_warshall<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        weights: &HashMap<EdgeId, f64>,
    ) -> Vec<Vec<f64>> {
        let nodes = graph.nodes();
        let n = nodes.len();
        let id_map: HashMap<NodeId, usize> = nodes.iter().enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut dist = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            dist[i][i] = 0.0;
        }

        for &eid in &graph.edges() {
            if let Some((from, to)) = graph.edge_endpoints(eid) {
                let w = weights.get(&eid).copied().unwrap_or(1.0);
                if let (Some(&i), Some(&j)) = (id_map.get(&from), id_map.get(&to)) {
                    dist[i][j] = dist[i][j].min(w);
                }
            }
        }

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        dist
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GraphColoring
// ═══════════════════════════════════════════════════════════════════════

/// Graph coloring algorithms.
pub struct GraphColoring;

impl GraphColoring {
    /// Greedy coloring.
    pub fn greedy<V: Clone, E: Clone>(graph: &Graph<V, E>) -> HashMap<NodeId, usize> {
        let mut colors: HashMap<NodeId, usize> = HashMap::new();
        let nodes = graph.nodes();

        for &node in &nodes {
            let neighbor_colors: HashSet<usize> = graph.successors(node).iter()
                .chain(graph.predecessors(node).iter())
                .filter_map(|n| colors.get(n).copied())
                .collect();

            let mut color = 0;
            while neighbor_colors.contains(&color) {
                color += 1;
            }
            colors.insert(node, color);
        }

        colors
    }

    /// Chromatic number upper bound.
    pub fn chromatic_number_bound<V: Clone, E: Clone>(graph: &Graph<V, E>) -> usize {
        let coloring = Self::greedy(graph);
        coloring.values().max().map(|&m| m + 1).unwrap_or(0)
    }

    /// Check if the graph is k-colorable (backtracking).
    pub fn is_k_colorable<V: Clone, E: Clone>(graph: &Graph<V, E>, k: usize) -> bool {
        let nodes = graph.nodes();
        let mut colors: HashMap<NodeId, usize> = HashMap::new();
        Self::color_backtrack(graph, &nodes, 0, k, &mut colors)
    }

    fn color_backtrack<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        nodes: &[NodeId],
        idx: usize,
        k: usize,
        colors: &mut HashMap<NodeId, usize>,
    ) -> bool {
        if idx == nodes.len() { return true; }
        let node = nodes[idx];

        for color in 0..k {
            let valid = graph.successors(node).iter()
                .chain(graph.predecessors(node).iter())
                .all(|n| colors.get(n) != Some(&color));
            if valid {
                colors.insert(node, color);
                if Self::color_backtrack(graph, nodes, idx + 1, k, colors) {
                    return true;
                }
                colors.remove(&node);
            }
        }
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DominatorTree
// ═══════════════════════════════════════════════════════════════════════

/// Dominator tree computation.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each node.
    pub idom: HashMap<NodeId, NodeId>,
    /// Entry node.
    pub entry: NodeId,
}

impl DominatorTree {
    /// Compute the dominator tree from an entry node.
    pub fn compute<V: Clone, E: Clone>(graph: &Graph<V, E>, entry: NodeId) -> Self {
        let nodes = graph.nodes();
        let n = nodes.len();
        let node_idx: HashMap<NodeId, usize> = nodes.iter().enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut doms: Vec<Option<usize>> = vec![None; n];
        let entry_idx = node_idx[&entry];
        doms[entry_idx] = Some(entry_idx);

        let rpo = Self::reverse_postorder(graph, entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &node in &rpo {
                if node == entry { continue; }
                let ni = node_idx[&node];
                let preds = graph.predecessors(node);
                let mut new_idom: Option<usize> = None;

                for &pred in &preds {
                    let pi = node_idx[&pred];
                    if doms[pi].is_some() {
                        new_idom = match new_idom {
                            None => Some(pi),
                            Some(cur) => Some(Self::intersect(&doms, cur, pi)),
                        };
                    }
                }

                if new_idom != doms[ni] {
                    doms[ni] = new_idom;
                    changed = true;
                }
            }
        }

        let mut idom = HashMap::new();
        for (i, dom) in doms.iter().enumerate() {
            if let Some(d) = dom {
                if *d != i {
                    idom.insert(nodes[i], nodes[*d]);
                }
            }
        }

        DominatorTree { idom, entry }
    }

    fn intersect(doms: &[Option<usize>], mut a: usize, mut b: usize) -> usize {
        while a != b {
            while a > b {
                a = doms[a].unwrap_or(a);
            }
            while b > a {
                b = doms[b].unwrap_or(b);
            }
        }
        a
    }

    fn reverse_postorder<V: Clone, E: Clone>(graph: &Graph<V, E>, entry: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        Self::rpo_dfs(graph, entry, &mut visited, &mut order);
        order.reverse();
        order
    }

    fn rpo_dfs<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) {
        if !visited.insert(node) { return; }
        for succ in graph.successors(node) {
            Self::rpo_dfs(graph, succ, visited, order);
        }
        order.push(node);
    }

    /// Check if a dominates b.
    pub fn dominates(&self, a: NodeId, b: NodeId) -> bool {
        if a == b { return true; }
        let mut current = b;
        while let Some(&dom) = self.idom.get(&current) {
            if dom == a { return true; }
            if dom == current { break; }
            current = dom;
        }
        false
    }

    /// Get the immediate dominator.
    pub fn immediate_dominator(&self, node: NodeId) -> Option<NodeId> {
        self.idom.get(&node).copied()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GraphTraversal — BFS and DFS iterators
// ═══════════════════════════════════════════════════════════════════════

/// Graph traversal algorithms.
pub struct GraphTraversal;

/// Edge classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Tree,
    Back,
    Forward,
    Cross,
}

impl GraphTraversal {
    /// BFS traversal.
    pub fn bfs<V: Clone, E: Clone>(graph: &Graph<V, E>, start: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            result.push(node);
            for succ in graph.successors(node) {
                if visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }

        result
    }

    /// DFS preorder traversal.
    pub fn dfs_preorder<V: Clone, E: Clone>(graph: &Graph<V, E>, start: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        Self::dfs_pre_visit(graph, start, &mut visited, &mut result);
        result
    }

    fn dfs_pre_visit<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) {
        if !visited.insert(node) { return; }
        result.push(node);
        for succ in graph.successors(node) {
            Self::dfs_pre_visit(graph, succ, visited, result);
        }
    }

    /// DFS postorder traversal.
    pub fn dfs_postorder<V: Clone, E: Clone>(graph: &Graph<V, E>, start: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        Self::dfs_post_visit(graph, start, &mut visited, &mut result);
        result
    }

    fn dfs_post_visit<V: Clone, E: Clone>(
        graph: &Graph<V, E>,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) {
        if !visited.insert(node) { return; }
        for succ in graph.successors(node) {
            Self::dfs_post_visit(graph, succ, visited, result);
        }
        result.push(node);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConnectedComponents
// ═══════════════════════════════════════════════════════════════════════

/// Connected component algorithms.
pub struct ConnectedComponents;

impl ConnectedComponents {
    /// Weakly connected components of a directed graph.
    pub fn weakly_connected<V: Clone, E: Clone>(graph: &Graph<V, E>) -> Vec<Vec<NodeId>> {
        let nodes = graph.nodes();
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in &nodes {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                visited.insert(node);
                queue.push_back(node);

                while let Some(n) = queue.pop_front() {
                    component.push(n);
                    for succ in graph.successors(n) {
                        if visited.insert(succ) {
                            queue.push_back(succ);
                        }
                    }
                    for pred in graph.predecessors(n) {
                        if visited.insert(pred) {
                            queue.push_back(pred);
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Check if graph is weakly connected.
    pub fn is_weakly_connected<V: Clone, E: Clone>(graph: &Graph<V, E>) -> bool {
        Self::weakly_connected(graph).len() <= 1
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GraphIsomorphism — basic graph isomorphism
// ═══════════════════════════════════════════════════════════════════════

/// Basic graph isomorphism checking.
pub struct GraphIsomorphism;

impl GraphIsomorphism {
    /// Check if two graphs are isomorphic (small graphs only).
    pub fn is_isomorphic<V: Clone, E: Clone>(
        g1: &Graph<V, E>,
        g2: &Graph<V, E>,
    ) -> bool {
        if g1.node_count() != g2.node_count() || g1.edge_count() != g2.edge_count() {
            return false;
        }
        Self::find_isomorphism(g1, g2).is_some()
    }

    /// Find an isomorphism mapping if one exists.
    pub fn find_isomorphism<V: Clone, E: Clone>(
        g1: &Graph<V, E>,
        g2: &Graph<V, E>,
    ) -> Option<HashMap<NodeId, NodeId>> {
        let nodes1 = g1.nodes();
        let nodes2 = g2.nodes();

        if nodes1.len() != nodes2.len() {
            return None;
        }

        // Degree-based pruning
        let mut deg1: Vec<(usize, usize, NodeId)> = nodes1.iter()
            .map(|&n| (g1.out_degree(n), g1.in_degree(n), n))
            .collect();
        let mut deg2: Vec<(usize, usize, NodeId)> = nodes2.iter()
            .map(|&n| (g2.out_degree(n), g2.in_degree(n), n))
            .collect();

        deg1.sort_by_key(|&(o, i, _)| (o, i));
        deg2.sort_by_key(|&(o, i, _)| (o, i));

        // Check degree sequences match
        if deg1.iter().map(|&(o, i, _)| (o, i)).collect::<Vec<_>>()
            != deg2.iter().map(|&(o, i, _)| (o, i)).collect::<Vec<_>>()
        {
            return None;
        }

        // Backtracking search
        let mut mapping: HashMap<NodeId, NodeId> = HashMap::new();
        let mut used: HashSet<NodeId> = HashSet::new();

        if Self::backtrack(g1, g2, &nodes1, &nodes2, 0, &mut mapping, &mut used) {
            Some(mapping)
        } else {
            None
        }
    }

    fn backtrack<V: Clone, E: Clone>(
        g1: &Graph<V, E>,
        g2: &Graph<V, E>,
        nodes1: &[NodeId],
        nodes2: &[NodeId],
        idx: usize,
        mapping: &mut HashMap<NodeId, NodeId>,
        used: &mut HashSet<NodeId>,
    ) -> bool {
        if idx == nodes1.len() {
            return true;
        }

        let n1 = nodes1[idx];
        for &n2 in nodes2 {
            if used.contains(&n2) { continue; }

            // Degree check
            if g1.out_degree(n1) != g2.out_degree(n2) { continue; }
            if g1.in_degree(n1) != g2.in_degree(n2) { continue; }

            // Consistency check
            let consistent = g1.successors(n1).iter().all(|&succ1| {
                if let Some(&succ2) = mapping.get(&succ1) {
                    g2.has_edge(n2, succ2)
                } else {
                    true
                }
            }) && g1.predecessors(n1).iter().all(|&pred1| {
                if let Some(&pred2) = mapping.get(&pred1) {
                    g2.has_edge(pred2, n2)
                } else {
                    true
                }
            });

            if consistent {
                mapping.insert(n1, n2);
                used.insert(n2);
                if Self::backtrack(g1, g2, nodes1, nodes2, idx + 1, mapping, used) {
                    return true;
                }
                mapping.remove(&n1);
                used.remove(&n2);
            }
        }

        false
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dag() -> Graph<&'static str, ()> {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        g.add_edge(a, b, ());
        g.add_edge(a, c, ());
        g.add_edge(b, d, ());
        g.add_edge(c, d, ());
        g
    }

    fn cycle_graph() -> Graph<&'static str, ()> {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge(a, b, ());
        g.add_edge(b, c, ());
        g.add_edge(c, a, ());
        g
    }

    #[test]
    fn test_graph_basic() {
        let g = simple_dag();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_graph_successors() {
        let g = simple_dag();
        let nodes = g.nodes();
        let succs = g.successors(nodes[0]);
        assert_eq!(succs.len(), 2);
    }

    #[test]
    fn test_graph_predecessors() {
        let g = simple_dag();
        let nodes = g.nodes();
        let preds = g.predecessors(nodes[3]); // D
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_topological_sort_kahn() {
        let g = simple_dag();
        let result = TopologicalSort::kahn(&g);
        assert!(result.is_ok());
        let order = result.unwrap();
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_dfs() {
        let g = simple_dag();
        let result = TopologicalSort::dfs(&g);
        assert!(result.is_ok());
        let order = result.unwrap();
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_cycle() {
        let g = cycle_graph();
        assert!(TopologicalSort::kahn(&g).is_err());
    }

    #[test]
    fn test_tarjan_scc_dag() {
        let g = simple_dag();
        let result = StronglyConnectedComponents::tarjan(&g);
        // DAG: each node is its own SCC
        assert_eq!(result.components.len(), 4);
    }

    #[test]
    fn test_tarjan_scc_cycle() {
        let g = cycle_graph();
        let result = StronglyConnectedComponents::tarjan(&g);
        // All 3 nodes in one SCC
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0].len(), 3);
    }

    #[test]
    fn test_kosaraju_scc() {
        let g = cycle_graph();
        let result = StronglyConnectedComponents::kosaraju(&g);
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0].len(), 3);
    }

    #[test]
    fn test_cycle_detection() {
        let dag = simple_dag();
        assert!(!CycleDetection::has_cycle(&dag));

        let cyclic = cycle_graph();
        assert!(CycleDetection::has_cycle(&cyclic));
    }

    #[test]
    fn test_find_cycle() {
        let g = cycle_graph();
        let cycle = CycleDetection::find_cycle(&g);
        assert!(cycle.is_some());
        assert!(cycle.unwrap().len() >= 2);
    }

    #[test]
    fn test_enumerate_cycles() {
        let g = cycle_graph();
        let cycles = CycleDetection::enumerate_cycles(&g, 5);
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_bfs_shortest_path() {
        let g = simple_dag();
        let nodes = g.nodes();
        let path = ShortestPath::bfs(&g, nodes[0], nodes[3]);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path[0], nodes[0]);
        assert_eq!(*path.last().unwrap(), nodes[3]);
    }

    #[test]
    fn test_bfs_no_path() {
        let mut g: Graph<(), ()> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        // No edge from a to b
        let path = ShortestPath::bfs(&g, a, b);
        assert!(path.is_none());
    }

    #[test]
    fn test_greedy_coloring() {
        let g = simple_dag();
        let coloring = GraphColoring::greedy(&g);
        assert_eq!(coloring.len(), 4);

        // Check no adjacent nodes have same color
        for &node in &g.nodes() {
            for succ in g.successors(node) {
                assert_ne!(coloring[&node], coloring[&succ]);
            }
        }
    }

    #[test]
    fn test_k_colorable() {
        let g = cycle_graph();
        assert!(!GraphColoring::is_k_colorable(&g, 1));
        assert!(GraphColoring::is_k_colorable(&g, 3));
    }

    #[test]
    fn test_dominator_tree() {
        let g = simple_dag();
        let nodes = g.nodes();
        let dom = DominatorTree::compute(&g, nodes[0]);
        assert!(dom.dominates(nodes[0], nodes[3]));
    }

    #[test]
    fn test_bfs_traversal() {
        let g = simple_dag();
        let nodes = g.nodes();
        let bfs = GraphTraversal::bfs(&g, nodes[0]);
        assert_eq!(bfs.len(), 4);
        assert_eq!(bfs[0], nodes[0]);
    }

    #[test]
    fn test_dfs_preorder() {
        let g = simple_dag();
        let nodes = g.nodes();
        let pre = GraphTraversal::dfs_preorder(&g, nodes[0]);
        assert_eq!(pre.len(), 4);
        assert_eq!(pre[0], nodes[0]);
    }

    #[test]
    fn test_weakly_connected() {
        let g = simple_dag();
        let comps = ConnectedComponents::weakly_connected(&g);
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn test_weakly_connected_disconnected() {
        let mut g: Graph<(), ()> = Graph::new();
        g.add_node(());
        g.add_node(());
        let comps = ConnectedComponents::weakly_connected(&g);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_graph_isomorphism() {
        // Two identical triangles
        let mut g1: Graph<(), ()> = Graph::new();
        let a1 = g1.add_node(());
        let b1 = g1.add_node(());
        let c1 = g1.add_node(());
        g1.add_edge(a1, b1, ());
        g1.add_edge(b1, c1, ());
        g1.add_edge(c1, a1, ());

        let mut g2: Graph<(), ()> = Graph::new();
        let a2 = g2.add_node(());
        let b2 = g2.add_node(());
        let c2 = g2.add_node(());
        g2.add_edge(a2, b2, ());
        g2.add_edge(b2, c2, ());
        g2.add_edge(c2, a2, ());

        assert!(GraphIsomorphism::is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_graph_not_isomorphic() {
        let mut g1: Graph<(), ()> = Graph::new();
        let a1 = g1.add_node(());
        let b1 = g1.add_node(());
        g1.add_edge(a1, b1, ());

        let mut g2: Graph<(), ()> = Graph::new();
        let a2 = g2.add_node(());
        let b2 = g2.add_node(());
        g2.add_edge(b2, a2, ());
        g2.add_edge(a2, b2, ());

        assert!(!GraphIsomorphism::is_isomorphic(&g1, &g2));
    }

    #[test]
    fn test_adjacency_matrix() {
        let g = simple_dag();
        let matrix = g.to_adjacency_matrix();
        assert_eq!(matrix.len(), 4);

        let g2 = from_adjacency_matrix(&matrix);
        assert_eq!(g2.node_count(), 4);
    }

    #[test]
    fn test_graph_density() {
        let g = simple_dag();
        let d = g.density();
        // 4 edges, 4 nodes: density = 4 / (4 * 3) = 1/3
        assert!((d - 1.0/3.0).abs() < 0.01);
    }
}
