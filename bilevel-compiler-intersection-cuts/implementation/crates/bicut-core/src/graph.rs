//! Dependency graph analysis using petgraph.
//!
//! Builds variable-constraint bipartite graphs, detects block structure,
//! performs decomposability analysis, and computes connected components
//! and sparsity patterns.

use bicut_types::{BilevelProblem, SparseMatrix, SparseMatrixCsr, DEFAULT_TOLERANCE};
use log::debug;
use petgraph::algo::connected_components;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Node type in the bipartite variable-constraint graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BipartiteNode {
    Variable(usize),
    Constraint(usize),
}

/// Edge in the bipartite graph (with coefficient weight).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BipartiteEdge {
    pub variable: usize,
    pub constraint: usize,
    pub coefficient: f64,
}

/// Block decomposition of a problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDecomposition {
    pub num_blocks: usize,
    pub blocks: Vec<Block>,
    pub is_decomposable: bool,
    pub linking_constraints: Vec<usize>,
}

/// A single block in the decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: usize,
    pub variable_indices: Vec<usize>,
    pub constraint_indices: Vec<usize>,
    pub size: usize,
}

/// Sparsity pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityPattern {
    pub row_nnz: Vec<usize>,
    pub col_nnz: Vec<usize>,
    pub max_row_nnz: usize,
    pub max_col_nnz: usize,
    pub avg_row_nnz: f64,
    pub avg_col_nnz: f64,
    pub bandwidth: usize,
    pub profile: usize,
}

/// Graph analysis summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysis {
    pub num_var_nodes: usize,
    pub num_constr_nodes: usize,
    pub num_edges: usize,
    pub num_connected_components: usize,
    pub block_decomposition: BlockDecomposition,
    pub sparsity: SparsityPattern,
    pub variable_degrees: Vec<usize>,
    pub constraint_degrees: Vec<usize>,
    pub is_biconnected: bool,
}

// ---------------------------------------------------------------------------
// DependencyGraph builder
// ---------------------------------------------------------------------------

/// Dependency graph for bilevel problem analysis.
pub struct DependencyGraph {
    graph: UnGraph<BipartiteNode, f64>,
    var_nodes: HashMap<usize, NodeIndex>,
    constr_nodes: HashMap<usize, NodeIndex>,
    num_vars: usize,
    num_constrs: usize,
}

impl DependencyGraph {
    /// Build the dependency graph from a sparse matrix.
    pub fn from_sparse_matrix(sm: &SparseMatrix, nrows: usize, ncols: usize) -> Self {
        let mut graph = UnGraph::new_undirected();
        let mut var_nodes = HashMap::new();
        let mut constr_nodes = HashMap::new();

        // Add variable nodes
        for j in 0..ncols {
            let idx = graph.add_node(BipartiteNode::Variable(j));
            var_nodes.insert(j, idx);
        }

        // Add constraint nodes
        for i in 0..nrows {
            let idx = graph.add_node(BipartiteNode::Constraint(i));
            constr_nodes.insert(i, idx);
        }

        // Add edges
        for entry in &sm.entries {
            if entry.value.abs() > DEFAULT_TOLERANCE {
                if let (Some(&var_node), Some(&constr_node)) =
                    (var_nodes.get(&entry.col), constr_nodes.get(&entry.row))
                {
                    graph.add_edge(var_node, constr_node, entry.value);
                }
            }
        }

        Self {
            graph,
            var_nodes,
            constr_nodes,
            num_vars: ncols,
            num_constrs: nrows,
        }
    }

    /// Build the dependency graph from a bilevel problem (lower-level only).
    pub fn from_bilevel_lower(problem: &BilevelProblem) -> Self {
        Self::from_sparse_matrix(
            &problem.lower_a,
            problem.num_lower_constraints,
            problem.num_lower_vars,
        )
    }

    /// Build a combined graph including both lower-level and linking matrices.
    pub fn from_bilevel_combined(problem: &BilevelProblem) -> Self {
        let m = problem.num_lower_constraints;
        let n_lower = problem.num_lower_vars;
        let n_upper = problem.num_upper_vars;
        let total_vars = n_upper + n_lower;

        let mut graph = UnGraph::new_undirected();
        let mut var_nodes = HashMap::new();
        let mut constr_nodes = HashMap::new();

        for j in 0..total_vars {
            let idx = graph.add_node(BipartiteNode::Variable(j));
            var_nodes.insert(j, idx);
        }

        for i in 0..m {
            let idx = graph.add_node(BipartiteNode::Constraint(i));
            constr_nodes.insert(i, idx);
        }

        // Lower-level constraint matrix: A*y (follower vars start at n_upper)
        for entry in &problem.lower_a.entries {
            if entry.value.abs() > DEFAULT_TOLERANCE {
                let var_idx = n_upper + entry.col; // follower variables
                if let (Some(&vn), Some(&cn)) =
                    (var_nodes.get(&var_idx), constr_nodes.get(&entry.row))
                {
                    graph.add_edge(vn, cn, entry.value);
                }
            }
        }

        // Linking matrix: B*x (leader vars start at 0)
        for entry in &problem.lower_linking_b.entries {
            if entry.value.abs() > DEFAULT_TOLERANCE {
                if let (Some(&vn), Some(&cn)) =
                    (var_nodes.get(&entry.col), constr_nodes.get(&entry.row))
                {
                    graph.add_edge(vn, cn, entry.value);
                }
            }
        }

        Self {
            graph,
            var_nodes,
            constr_nodes,
            num_vars: total_vars,
            num_constrs: m,
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of connected components.
    pub fn num_components(&self) -> usize {
        connected_components(&self.graph)
    }

    /// Compute the degree of each variable node.
    pub fn variable_degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0; self.num_vars];
        for (&var_idx, &node_idx) in &self.var_nodes {
            degrees[var_idx] = self.graph.edges(node_idx).count();
        }
        degrees
    }

    /// Compute the degree of each constraint node.
    pub fn constraint_degrees(&self) -> Vec<usize> {
        let mut degrees = vec![0; self.num_constrs];
        for (&constr_idx, &node_idx) in &self.constr_nodes {
            degrees[constr_idx] = self.graph.edges(node_idx).count();
        }
        degrees
    }

    /// Detect block structure via connected components in the bipartite graph.
    pub fn detect_blocks(&self) -> BlockDecomposition {
        let n_components = connected_components(&self.graph);

        if n_components <= 1 {
            let var_indices: Vec<usize> = (0..self.num_vars).collect();
            let constr_indices: Vec<usize> = (0..self.num_constrs).collect();
            let size = var_indices.len() + constr_indices.len();
            return BlockDecomposition {
                num_blocks: 1,
                blocks: vec![Block {
                    id: 0,
                    variable_indices: var_indices,
                    constraint_indices: constr_indices,
                    size,
                }],
                is_decomposable: false,
                linking_constraints: vec![],
            };
        }

        // Assign each node to its component via BFS
        let mut component_map: HashMap<NodeIndex, usize> = HashMap::new();
        let mut visited = HashSet::new();
        let mut comp_id = 0;

        for node in self.graph.node_indices() {
            if visited.contains(&node) {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                component_map.insert(current, comp_id);
                for neighbor in self.graph.neighbors(current) {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            comp_id += 1;
        }

        // Build blocks
        let mut blocks = Vec::new();
        for c in 0..comp_id {
            let mut var_indices = Vec::new();
            let mut constr_indices = Vec::new();

            for (&var_idx, &node_idx) in &self.var_nodes {
                if component_map.get(&node_idx) == Some(&c) {
                    var_indices.push(var_idx);
                }
            }
            for (&constr_idx, &node_idx) in &self.constr_nodes {
                if component_map.get(&node_idx) == Some(&c) {
                    constr_indices.push(constr_idx);
                }
            }

            var_indices.sort_unstable();
            constr_indices.sort_unstable();

            if !var_indices.is_empty() || !constr_indices.is_empty() {
                let size = var_indices.len() + constr_indices.len();
                blocks.push(Block {
                    id: blocks.len(),
                    variable_indices: var_indices,
                    constraint_indices: constr_indices,
                    size,
                });
            }
        }

        blocks.sort_by(|a, b| b.size.cmp(&a.size));

        BlockDecomposition {
            num_blocks: blocks.len(),
            blocks,
            is_decomposable: comp_id > 1,
            linking_constraints: vec![],
        }
    }

    /// Compute the sparsity pattern of the underlying matrix.
    pub fn compute_sparsity_pattern(&self) -> SparsityPattern {
        let var_deg = self.variable_degrees();
        let constr_deg = self.constraint_degrees();

        let max_row_nnz = constr_deg.iter().cloned().max().unwrap_or(0);
        let max_col_nnz = var_deg.iter().cloned().max().unwrap_or(0);

        let avg_row_nnz = if self.num_constrs > 0 {
            constr_deg.iter().sum::<usize>() as f64 / self.num_constrs as f64
        } else {
            0.0
        };

        let avg_col_nnz = if self.num_vars > 0 {
            var_deg.iter().sum::<usize>() as f64 / self.num_vars as f64
        } else {
            0.0
        };

        // Bandwidth: max |i - j| for all edges
        let mut bandwidth = 0usize;
        let mut profile = 0usize;

        for edge in self.graph.edge_references() {
            let src = edge.source();
            let tgt = edge.target();
            if let (Some(BipartiteNode::Variable(j)), Some(BipartiteNode::Constraint(i))) =
                (self.graph.node_weight(src), self.graph.node_weight(tgt))
            {
                let diff = if *i > *j { *i - *j } else { *j - *i };
                if diff > bandwidth {
                    bandwidth = diff;
                }
                profile += diff;
            } else if let (Some(BipartiteNode::Constraint(i)), Some(BipartiteNode::Variable(j))) =
                (self.graph.node_weight(src), self.graph.node_weight(tgt))
            {
                let diff = if *i > *j { *i - *j } else { *j - *i };
                if diff > bandwidth {
                    bandwidth = diff;
                }
                profile += diff;
            }
        }

        SparsityPattern {
            row_nnz: constr_deg.clone(),
            col_nnz: var_deg.clone(),
            max_row_nnz,
            max_col_nnz,
            avg_row_nnz,
            avg_col_nnz,
            bandwidth,
            profile,
        }
    }

    /// Perform full graph analysis.
    pub fn analyze(&self) -> GraphAnalysis {
        let block_decomposition = self.detect_blocks();
        let sparsity = self.compute_sparsity_pattern();
        let var_deg = self.variable_degrees();
        let constr_deg = self.constraint_degrees();
        let n_components = self.num_components();

        // Simple biconnectivity check: a graph is biconnected if removing any
        // single node keeps it connected. For our purposes, approximate by
        // checking if all degrees > 1.
        let is_biconnected = if self.graph.node_count() < 3 {
            false
        } else {
            let min_degree = self
                .graph
                .node_indices()
                .map(|n| self.graph.edges(n).count())
                .min()
                .unwrap_or(0);
            min_degree >= 2 && n_components == 1
        };

        GraphAnalysis {
            num_var_nodes: self.num_vars,
            num_constr_nodes: self.num_constrs,
            num_edges: self.edge_count(),
            num_connected_components: n_components,
            block_decomposition,
            sparsity,
            variable_degrees: var_deg,
            constraint_degrees: constr_deg,
            is_biconnected,
        }
    }

    /// Find isolated variables (degree 0 in the constraint graph).
    pub fn isolated_variables(&self) -> Vec<usize> {
        let degrees = self.variable_degrees();
        degrees
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find isolated constraints (degree 0 in the variable graph).
    pub fn isolated_constraints(&self) -> Vec<usize> {
        let degrees = self.constraint_degrees();
        degrees
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find articulation points (nodes whose removal disconnects the graph).
    pub fn articulation_points(&self) -> Vec<BipartiteNode> {
        let base_components = self.num_components();
        let mut points = Vec::new();

        // Simple O(V * (V+E)) approach: remove each node and count components
        for node_idx in self.graph.node_indices() {
            let weight = *self.graph.node_weight(node_idx).unwrap();
            // Count components without this node
            let neighbors: Vec<NodeIndex> = self.graph.neighbors(node_idx).collect();
            if neighbors.is_empty() {
                continue;
            }

            // Use union-find on remaining nodes
            let all_nodes: Vec<NodeIndex> = self
                .graph
                .node_indices()
                .filter(|&n| n != node_idx)
                .collect();

            if all_nodes.is_empty() {
                continue;
            }

            let mut node_to_idx: HashMap<NodeIndex, usize> = HashMap::new();
            for (i, &n) in all_nodes.iter().enumerate() {
                node_to_idx.insert(n, i);
            }

            let mut uf = SimpleUnionFind::new(all_nodes.len());
            for &n in &all_nodes {
                for neighbor in self.graph.neighbors(n) {
                    if neighbor != node_idx {
                        if let (Some(&a), Some(&b)) =
                            (node_to_idx.get(&n), node_to_idx.get(&neighbor))
                        {
                            uf.union(a, b);
                        }
                    }
                }
            }

            let components: HashSet<usize> = (0..all_nodes.len()).map(|i| uf.find(i)).collect();

            if components.len() > base_components {
                points.push(weight);
            }
        }

        points
    }
}

struct SimpleUnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl SimpleUnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_block_diagonal_matrix() -> SparseMatrix {
        let mut sm = SparseMatrix::new(4, 4);
        // Block 1: rows 0,1 x cols 0,1
        sm.add_entry(0, 0, 1.0);
        sm.add_entry(0, 1, 2.0);
        sm.add_entry(1, 0, 3.0);
        sm.add_entry(1, 1, 4.0);
        // Block 2: rows 2,3 x cols 2,3
        sm.add_entry(2, 2, 5.0);
        sm.add_entry(2, 3, 6.0);
        sm.add_entry(3, 2, 7.0);
        sm.add_entry(3, 3, 8.0);
        sm
    }

    fn make_connected_matrix() -> SparseMatrix {
        let mut sm = SparseMatrix::new(3, 3);
        sm.add_entry(0, 0, 1.0);
        sm.add_entry(0, 1, 1.0);
        sm.add_entry(1, 1, 1.0);
        sm.add_entry(1, 2, 1.0);
        sm.add_entry(2, 0, 1.0);
        sm.add_entry(2, 2, 1.0);
        sm
    }

    fn make_test_bilevel() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 3);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 1.0);
        lower_a.add_entry(1, 1, 1.0);
        lower_a.add_entry(1, 2, 1.0);

        let mut linking = SparseMatrix::new(2, 1);
        linking.add_entry(0, 0, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: linking,
            upper_constraints_a: SparseMatrix::new(0, 4),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 3,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_block_diagonal_detection() {
        let sm = make_block_diagonal_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 4, 4);
        let blocks = dg.detect_blocks();
        assert_eq!(blocks.num_blocks, 2);
        assert!(blocks.is_decomposable);
    }

    #[test]
    fn test_connected_graph() {
        let sm = make_connected_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 3, 3);
        assert_eq!(dg.num_components(), 1);
    }

    #[test]
    fn test_variable_degrees() {
        let sm = make_connected_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 3, 3);
        let degrees = dg.variable_degrees();
        assert_eq!(degrees[0], 2); // col 0 appears in rows 0, 2
        assert_eq!(degrees[1], 2); // col 1 appears in rows 0, 1
    }

    #[test]
    fn test_sparsity_pattern() {
        let sm = make_connected_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 3, 3);
        let sp = dg.compute_sparsity_pattern();
        assert_eq!(sp.max_row_nnz, 2);
    }

    #[test]
    fn test_bilevel_lower_graph() {
        let p = make_test_bilevel();
        let dg = DependencyGraph::from_bilevel_lower(&p);
        assert_eq!(dg.node_count(), 2 + 3); // 2 constraints + 3 variables
    }

    #[test]
    fn test_combined_graph() {
        let p = make_test_bilevel();
        let dg = DependencyGraph::from_bilevel_combined(&p);
        // 1 leader + 3 follower vars + 2 constraints = 6 nodes
        assert_eq!(dg.node_count(), 4 + 2);
    }

    #[test]
    fn test_full_analysis() {
        let sm = make_connected_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 3, 3);
        let analysis = dg.analyze();
        assert_eq!(analysis.num_connected_components, 1);
        assert!(!analysis.block_decomposition.is_decomposable);
    }

    #[test]
    fn test_isolated_variables() {
        let mut sm = SparseMatrix::new(2, 3);
        sm.add_entry(0, 0, 1.0);
        sm.add_entry(1, 1, 1.0);
        // Variable 2 is isolated
        let dg = DependencyGraph::from_sparse_matrix(&sm, 2, 3);
        let isolated = dg.isolated_variables();
        assert!(isolated.contains(&2));
    }

    #[test]
    fn test_edge_count() {
        let sm = make_connected_matrix();
        let dg = DependencyGraph::from_sparse_matrix(&sm, 3, 3);
        assert_eq!(dg.edge_count(), 6);
    }
}
