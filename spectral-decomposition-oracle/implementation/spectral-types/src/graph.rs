//! Hypergraph and ordinary graph data structures for constraint structure analysis.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use crate::sparse::CsrMatrix;

/// A hypergraph where each hyperedge connects an arbitrary set of vertices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypergraph {
    pub num_vertices: usize,
    pub edges: Vec<Vec<usize>>,
    pub edge_weights: Vec<f64>,
    pub vertex_labels: Vec<String>,
}

impl Hypergraph {
    pub fn new(num_vertices: usize) -> Self {
        Self {
            num_vertices,
            edges: Vec::new(),
            edge_weights: Vec::new(),
            vertex_labels: (0..num_vertices).map(|i| format!("v{}", i)).collect(),
        }
    }

    pub fn add_edge(&mut self, vertices: Vec<usize>, weight: f64) {
        self.edges.push(vertices);
        self.edge_weights.push(weight);
    }

    pub fn num_edges(&self) -> usize { self.edges.len() }

    pub fn from_constraint_matrix(matrix: &CsrMatrix<f64>) -> Self {
        let mut hg = Self::new(matrix.cols);
        for i in 0..matrix.rows {
            let cols: Vec<usize> = matrix.row_indices(i).to_vec();
            if !cols.is_empty() {
                let weight = matrix.row_values(i).iter().map(|v| v.abs()).sum::<f64>().max(1.0);
                hg.add_edge(cols, weight);
            }
        }
        hg
    }

    pub fn vertex_degree(&self, v: usize) -> usize {
        self.edges.iter().filter(|e| e.contains(&v)).count()
    }

    pub fn vertex_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_vertices];
        for e in &self.edges {
            for &v in e { if v < self.num_vertices { deg[v] += 1; } }
        }
        deg
    }

    pub fn edge_sizes(&self) -> Vec<usize> {
        self.edges.iter().map(|e| e.len()).collect()
    }

    pub fn neighbors(&self, v: usize) -> BTreeSet<usize> {
        let mut nbrs = BTreeSet::new();
        for e in &self.edges {
            if e.contains(&v) {
                for &u in e { if u != v { nbrs.insert(u); } }
            }
        }
        nbrs
    }

    pub fn vertex_edge_incidence(&self, v: usize) -> Vec<usize> {
        self.edges.iter().enumerate()
            .filter(|(_, e)| e.contains(&v))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_vertices];
        let mut components = Vec::new();

        // Build adjacency from hyperedges
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.num_vertices];
        for e in &self.edges {
            for i in 0..e.len() {
                for j in (i + 1)..e.len() {
                    let u = e[i]; let v = e[j];
                    if u < self.num_vertices && v < self.num_vertices {
                        adj[u].push(v); adj[v].push(u);
                    }
                }
            }
        }

        for start in 0..self.num_vertices {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(u) = queue.pop_front() {
                comp.push(u);
                for &v in &adj[u] {
                    if !visited[v] { visited[v] = true; queue.push_back(v); }
                }
            }
            components.push(comp);
        }
        components
    }

    pub fn subgraph(&self, vertices: &HashSet<usize>) -> Self {
        let remap: HashMap<usize, usize> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut sg = Self::new(vertices.len());
        for (ei, e) in self.edges.iter().enumerate() {
            let mapped: Vec<usize> = e.iter().filter_map(|v| remap.get(v).copied()).collect();
            if mapped.len() >= 2 {
                sg.add_edge(mapped, self.edge_weights[ei]);
            }
        }
        sg
    }

    /// Clique expansion: create an ordinary graph where hyperedge vertices form cliques.
    pub fn clique_expansion(&self) -> Graph {
        let mut g = Graph::new(self.num_vertices);
        for (ei, e) in self.edges.iter().enumerate() {
            let w = self.edge_weights[ei];
            let edge_w = if e.len() > 1 { w / (e.len() - 1) as f64 } else { w };
            for i in 0..e.len() {
                for j in (i + 1)..e.len() {
                    g.add_edge(e[i], e[j], edge_w);
                }
            }
        }
        g
    }

    /// Build vertex-hyperedge incidence matrix (|V| x |E|).
    pub fn incidence_matrix(&self) -> CsrMatrix<f64> {
        let mut triplets = crate::sparse::SparseTriple::new(self.num_vertices, self.num_edges());
        for (j, e) in self.edges.iter().enumerate() {
            for &v in e {
                if v < self.num_vertices { triplets.add(v, j, 1.0); }
            }
        }
        triplets.to_csr()
    }

    /// Count edges crossing a partition (each block is a set of vertices).
    pub fn crossing_edges(&self, partition: &[usize]) -> usize {
        self.edges.iter().filter(|e| {
            let blocks: HashSet<usize> = e.iter().filter_map(|&v| partition.get(v).copied()).collect();
            blocks.len() > 1
        }).count()
    }

    pub fn total_edge_weight(&self) -> f64 {
        self.edge_weights.iter().sum()
    }
}

/// An ordinary weighted graph with adjacency list representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub num_vertices: usize,
    pub adjacency: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self { num_vertices: n, adjacency: vec![Vec::new(); n] }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        if u >= self.num_vertices || v >= self.num_vertices { return; }
        // Check if edge already exists and update weight
        if let Some(e) = self.adjacency[u].iter_mut().find(|e| e.0 == v) {
            e.1 += weight;
        } else {
            self.adjacency[u].push((v, weight));
        }
        if u != v {
            if let Some(e) = self.adjacency[v].iter_mut().find(|e| e.0 == u) {
                e.1 += weight;
            } else {
                self.adjacency[v].push((u, weight));
            }
        }
    }

    pub fn degree(&self, v: usize) -> usize {
        if v >= self.num_vertices { 0 } else { self.adjacency[v].len() }
    }

    pub fn weighted_degree(&self, v: usize) -> f64 {
        if v >= self.num_vertices { return 0.0; }
        self.adjacency[v].iter().map(|(_, w)| w).sum()
    }

    pub fn degrees(&self) -> Vec<usize> {
        (0..self.num_vertices).map(|v| self.adjacency[v].len()).collect()
    }

    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }

    pub fn neighbors(&self, v: usize) -> Vec<usize> {
        if v >= self.num_vertices { return Vec::new(); }
        self.adjacency[v].iter().map(|(u, _)| *u).collect()
    }

    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_vertices];
        let mut components = Vec::new();
        for start in 0..self.num_vertices {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(u) = queue.pop_front() {
                comp.push(u);
                for &(v, _) in &self.adjacency[u] {
                    if !visited[v] { visited[v] = true; queue.push_back(v); }
                }
            }
            components.push(comp);
        }
        components
    }

    pub fn subgraph(&self, vertices: &[usize]) -> Graph {
        let vset: HashSet<usize> = vertices.iter().copied().collect();
        let remap: HashMap<usize, usize> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut g = Graph::new(vertices.len());
        for &u in vertices {
            if u >= self.num_vertices { continue; }
            for &(v, w) in &self.adjacency[u] {
                if v > u && vset.contains(&v) {
                    g.add_edge(remap[&u], remap[&v], w);
                }
            }
        }
        g
    }

    /// Build the graph Laplacian as a dense matrix: L = D - A.
    pub fn laplacian_dense(&self) -> crate::dense::DenseMatrix<f64> {
        let n = self.num_vertices;
        let mut l = crate::dense::DenseMatrix::zeros(n, n);
        for u in 0..n {
            let mut diag = 0.0;
            for &(v, w) in &self.adjacency[u] {
                if u != v {
                    let _ = l.set(u, v, -w);
                    diag += w;
                }
            }
            let _ = l.set(u, u, diag);
        }
        l
    }

    /// Build the normalized Laplacian: I - D^{-1/2} A D^{-1/2}.
    pub fn normalized_laplacian_dense(&self) -> crate::dense::DenseMatrix<f64> {
        let n = self.num_vertices;
        let deg: Vec<f64> = (0..n).map(|v| self.weighted_degree(v)).collect();
        let inv_sqrt_deg: Vec<f64> = deg.iter().map(|&d| if d > 1e-15 { 1.0 / d.sqrt() } else { 0.0 }).collect();

        let mut l = crate::dense::DenseMatrix::zeros(n, n);
        for u in 0..n {
            if deg[u] > 1e-15 { let _ = l.set(u, u, 1.0); }
            for &(v, w) in &self.adjacency[u] {
                if u != v {
                    let val = -w * inv_sqrt_deg[u] * inv_sqrt_deg[v];
                    let _ = l.set(u, v, val);
                }
            }
        }
        l
    }

    /// Edge density: |E| / (|V| choose 2).
    pub fn density(&self) -> f64 {
        let n = self.num_vertices;
        if n < 2 { return 0.0; }
        let max_edges = n * (n - 1) / 2;
        self.num_edges() as f64 / max_edges as f64
    }

    pub fn total_weight(&self) -> f64 {
        self.adjacency.iter().flat_map(|adj| adj.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }

    /// Cut weight for a partition assignment.
    pub fn cut_weight(&self, partition: &[usize]) -> f64 {
        let mut cut = 0.0;
        for u in 0..self.num_vertices {
            for &(v, w) in &self.adjacency[u] {
                if v > u && partition.get(u) != partition.get(v) {
                    cut += w;
                }
            }
        }
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hypergraph() -> Hypergraph {
        let mut hg = Hypergraph::new(5);
        hg.add_edge(vec![0, 1, 2], 1.0);
        hg.add_edge(vec![2, 3], 1.0);
        hg.add_edge(vec![3, 4], 1.0);
        hg
    }

    #[test] fn test_hg_basic() {
        let hg = sample_hypergraph();
        assert_eq!(hg.num_vertices, 5);
        assert_eq!(hg.num_edges(), 3);
    }

    #[test] fn test_vertex_degrees() {
        let hg = sample_hypergraph();
        let deg = hg.vertex_degrees();
        assert_eq!(deg[0], 1); assert_eq!(deg[2], 2); assert_eq!(deg[3], 2);
    }

    #[test] fn test_neighbors() {
        let hg = sample_hypergraph();
        let nbrs = hg.neighbors(2);
        assert!(nbrs.contains(&0)); assert!(nbrs.contains(&1)); assert!(nbrs.contains(&3));
    }

    #[test] fn test_connected_components() {
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1], 1.0);
        hg.add_edge(vec![2, 3], 1.0);
        let cc = hg.connected_components();
        assert_eq!(cc.len(), 2);
    }

    #[test] fn test_clique_expansion() {
        let hg = sample_hypergraph();
        let g = hg.clique_expansion();
        assert_eq!(g.num_vertices, 5);
        assert!(g.degree(2) >= 2);
    }

    #[test] fn test_crossing_edges() {
        let hg = sample_hypergraph();
        let partition = vec![0, 0, 0, 1, 1];
        assert_eq!(hg.crossing_edges(&partition), 1); // edge {2,3} crosses
    }

    #[test] fn test_incidence_matrix() {
        let hg = sample_hypergraph();
        let inc = hg.incidence_matrix();
        assert_eq!(inc.shape(), (5, 3));
    }

    #[test] fn test_graph_basic() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 1.0);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.degree(1), 2);
    }

    #[test] fn test_graph_components() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        assert_eq!(g.connected_components().len(), 2);
    }

    #[test] fn test_graph_laplacian() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
        let l = g.laplacian_dense();
        assert_eq!(l.get(0, 0), Some(1.0));
        assert_eq!(l.get(1, 1), Some(2.0));
        assert_eq!(l.get(0, 1), Some(-1.0));
    }

    #[test] fn test_graph_cut_weight() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        let partition = vec![0, 0, 1, 1];
        assert!((g.cut_weight(&partition) - 2.0).abs() < 1e-10);
    }

    #[test] fn test_graph_density() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(0, 2, 1.0); g.add_edge(0, 3, 1.0);
        g.add_edge(1, 2, 1.0); g.add_edge(1, 3, 1.0); g.add_edge(2, 3, 1.0);
        assert!((g.density() - 1.0).abs() < 1e-10);
    }

    #[test] fn test_graph_subgraph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        let sg = g.subgraph(&[1, 2]);
        assert_eq!(sg.num_vertices, 2); assert_eq!(sg.num_edges(), 1);
    }

    #[test] fn test_hg_total_weight() {
        let hg = sample_hypergraph();
        assert!((hg.total_edge_weight() - 3.0).abs() < 1e-10);
    }
}
