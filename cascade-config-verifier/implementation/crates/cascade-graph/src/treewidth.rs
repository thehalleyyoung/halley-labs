//! Treewidth estimation using elimination-order heuristics.

use crate::rtig::RtigGraph;
use cascade_types::service::ServiceId;
use indexmap::IndexMap;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Bag & TreeDecomposition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bag {
    pub nodes: Vec<ServiceId>,
}

impl Bag {
    pub fn new(nodes: Vec<ServiceId>) -> Self { Self { nodes } }
    pub fn contains(&self, id: &ServiceId) -> bool { self.nodes.contains(id) }
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn intersection(&self, other: &Bag) -> Bag {
        let set: HashSet<&ServiceId> = self.nodes.iter().collect();
        Bag { nodes: other.nodes.iter().filter(|n| set.contains(n)).cloned().collect() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeDecomposition {
    pub bags: Vec<Bag>,
    pub tree_edges: Vec<(usize, usize)>,
    pub width: usize,
}

impl TreeDecomposition {
    pub fn new() -> Self { Self { bags: Vec::new(), tree_edges: Vec::new(), width: 0 } }

    pub fn add_bag(&mut self, bag: Bag) -> usize {
        let idx = self.bags.len();
        let w = if bag.len() > 0 { bag.len() - 1 } else { 0 };
        if w > self.width { self.width = w; }
        self.bags.push(bag);
        idx
    }

    pub fn add_tree_edge(&mut self, a: usize, b: usize) {
        self.tree_edges.push((a, b));
    }

    pub fn width(&self) -> usize { self.width }
    pub fn num_bags(&self) -> usize { self.bags.len() }

    pub fn validate(&self, graph: &RtigGraph) -> bool {
        // 1. Every vertex appears in at least one bag.
        let all_svcs = graph.services();
        for svc in &all_svcs {
            if !self.bags.iter().any(|b| b.contains(svc)) { return false; }
        }
        // 2. Every edge has both endpoints in some bag.
        for e in graph.inner.edge_references() {
            let s = graph.inner.node_weight(e.source()).unwrap();
            let t = graph.inner.node_weight(e.target()).unwrap();
            if !self.bags.iter().any(|b| b.contains(s) && b.contains(t)) { return false; }
        }
        // 3. For each vertex, the bags containing it form a connected subtree.
        let nb = self.bags.len();
        let mut tree_adj: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for &(a, b) in &self.tree_edges {
            if a < nb && b < nb {
                tree_adj[a].push(b);
                tree_adj[b].push(a);
            }
        }
        for svc in &all_svcs {
            let containing: Vec<usize> = (0..nb).filter(|&i| self.bags[i].contains(svc)).collect();
            if containing.is_empty() { continue; }
            // BFS to check connectivity in subtree.
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            visited.insert(containing[0]);
            queue.push_back(containing[0]);
            while let Some(v) = queue.pop_front() {
                for &nb_idx in &tree_adj[v] {
                    if !visited.contains(&nb_idx) && self.bags[nb_idx].contains(svc) {
                        visited.insert(nb_idx);
                        queue.push_back(nb_idx);
                    }
                }
            }
            if visited.len() != containing.len() { return false; }
        }
        true
    }
}

impl Default for TreeDecomposition { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// EstimationMethod & TreewidthEstimator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EstimationMethod {
    MinDegree,
    MinFill,
    MetisBased,
}

pub struct TreewidthEstimator;

impl TreewidthEstimator {
    pub fn new() -> Self { Self }

    pub fn estimate(&self, graph: &RtigGraph, method: EstimationMethod) -> usize {
        match method {
            EstimationMethod::MinDegree => compute_treewidth_upper_bound(graph),
            EstimationMethod::MinFill => compute_treewidth_upper_bound_min_fill(graph),
            EstimationMethod::MetisBased => compute_treewidth_upper_bound(graph), // fallback
        }
    }

    pub fn estimate_all_methods(&self, graph: &RtigGraph) -> IndexMap<EstimationMethod, usize> {
        let mut results = IndexMap::new();
        results.insert(EstimationMethod::MinDegree, compute_treewidth_upper_bound(graph));
        results.insert(EstimationMethod::MinFill, compute_treewidth_upper_bound_min_fill(graph));
        results.insert(EstimationMethod::MetisBased, compute_treewidth_upper_bound(graph));
        results
    }
}

impl Default for TreewidthEstimator { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Core elimination algorithms
// ---------------------------------------------------------------------------

fn build_undirected_adj(graph: &RtigGraph) -> (Vec<ServiceId>, Vec<HashSet<usize>>) {
    let ids: Vec<ServiceId> = graph.services();
    let n = ids.len();
    let idx_of: HashMap<ServiceId, usize> = ids.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for e in graph.inner.edge_references() {
        let s = graph.inner.node_weight(e.source()).unwrap();
        let t = graph.inner.node_weight(e.target()).unwrap();
        if let (Some(&u), Some(&v)) = (idx_of.get(s), idx_of.get(t)) {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
    (ids, adj)
}

/// Min-degree elimination heuristic.
pub fn compute_treewidth_upper_bound(graph: &RtigGraph) -> usize {
    let (ids, mut adj) = build_undirected_adj(graph);
    let n = ids.len();
    if n == 0 { return 0; }
    let mut eliminated = vec![false; n];
    let mut max_clique = 0usize;

    for _ in 0..n {
        let next = (0..n).filter(|&i| !eliminated[i])
            .min_by_key(|&i| adj[i].iter().filter(|&&j| !eliminated[j]).count());
        let v = match next { Some(v) => v, None => break };
        let nbrs: Vec<usize> = adj[v].iter().copied().filter(|&j| !eliminated[j]).collect();
        if nbrs.len() > max_clique { max_clique = nbrs.len(); }
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                adj[nbrs[a]].insert(nbrs[b]);
                adj[nbrs[b]].insert(nbrs[a]);
            }
        }
        eliminated[v] = true;
    }
    max_clique
}

/// Min-fill elimination heuristic.
pub fn compute_treewidth_upper_bound_min_fill(graph: &RtigGraph) -> usize {
    let (ids, mut adj) = build_undirected_adj(graph);
    let n = ids.len();
    if n == 0 { return 0; }
    let mut eliminated = vec![false; n];
    let mut max_clique = 0usize;

    for _ in 0..n {
        let next = (0..n).filter(|&i| !eliminated[i])
            .min_by_key(|&i| {
                let nbrs: Vec<usize> = adj[i].iter().copied().filter(|&j| !eliminated[j]).collect();
                let mut fill = 0usize;
                for a in 0..nbrs.len() {
                    for b in (a + 1)..nbrs.len() {
                        if !adj[nbrs[a]].contains(&nbrs[b]) { fill += 1; }
                    }
                }
                fill
            });
        let v = match next { Some(v) => v, None => break };
        let nbrs: Vec<usize> = adj[v].iter().copied().filter(|&j| !eliminated[j]).collect();
        if nbrs.len() > max_clique { max_clique = nbrs.len(); }
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                adj[nbrs[a]].insert(nbrs[b]);
                adj[nbrs[b]].insert(nbrs[a]);
            }
        }
        eliminated[v] = true;
    }
    max_clique
}

/// Build a tree decomposition using the min-degree elimination ordering.
pub fn greedy_tree_decomposition(graph: &RtigGraph) -> TreeDecomposition {
    let (ids, mut adj) = build_undirected_adj(graph);
    let n = ids.len();
    let mut td = TreeDecomposition::new();
    if n == 0 { return td; }
    let mut eliminated = vec![false; n];

    for _ in 0..n {
        let next = (0..n).filter(|&i| !eliminated[i])
            .min_by_key(|&i| adj[i].iter().filter(|&&j| !eliminated[j]).count());
        let v = match next { Some(v) => v, None => break };
        let nbrs: Vec<usize> = adj[v].iter().copied().filter(|&j| !eliminated[j]).collect();
        let mut bag_nodes = vec![ids[v].clone()];
        for &nb in &nbrs { bag_nodes.push(ids[nb].clone()); }
        let bag_idx = td.add_bag(Bag::new(bag_nodes));
        if bag_idx > 0 { td.add_tree_edge(bag_idx - 1, bag_idx); }
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                adj[nbrs[a]].insert(nbrs[b]);
                adj[nbrs[b]].insert(nbrs[a]);
            }
        }
        eliminated[v] = true;
    }
    td
}

/// Validate a tree decomposition against the original graph.
pub fn validate_tree_decomposition(graph: &RtigGraph, decomp: &TreeDecomposition) -> bool {
    decomp.validate(graph)
}

/// Convert to a nice tree decomposition where each bag differs from parent by one node.
pub fn to_nice_decomposition(decomp: &TreeDecomposition) -> TreeDecomposition {
    let mut nice = TreeDecomposition::new();
    if decomp.bags.is_empty() { return nice; }

    // For each consecutive pair of bags, insert intermediate introduce/forget bags.
    let first_idx = nice.add_bag(decomp.bags[0].clone());
    let mut prev_idx = first_idx;
    let prev_set: HashSet<ServiceId> = decomp.bags[0].nodes.iter().cloned().collect();
    let mut current_set = prev_set;

    for i in 1..decomp.bags.len() {
        let target_set: HashSet<ServiceId> = decomp.bags[i].nodes.iter().cloned().collect();
        // Forget nodes not in target.
        let to_forget: Vec<ServiceId> = current_set.difference(&target_set).cloned().collect();
        for node in to_forget {
            current_set.remove(&node);
            let bag = Bag::new(current_set.iter().cloned().collect());
            let idx = nice.add_bag(bag);
            nice.add_tree_edge(prev_idx, idx);
            prev_idx = idx;
        }
        // Introduce nodes in target but not current.
        let to_introduce: Vec<ServiceId> = target_set.difference(&current_set).cloned().collect();
        for node in to_introduce {
            current_set.insert(node);
            let bag = Bag::new(current_set.iter().cloned().collect());
            let idx = nice.add_bag(bag);
            nice.add_tree_edge(prev_idx, idx);
            prev_idx = idx;
        }
    }
    nice
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtig::{build_chain, build_diamond, RtigGraph};
    use cascade_types::policy::ResiliencePolicy;
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId { ServiceId::new(s) }

    #[test]
    fn test_empty_graph() {
        let g = RtigGraph::new();
        assert_eq!(compute_treewidth_upper_bound(&g), 0);
    }

    #[test]
    fn test_single_edge() {
        let g = build_chain(&["A","B"], 1);
        assert_eq!(compute_treewidth_upper_bound(&g), 1);
    }

    #[test]
    fn test_path_graph() {
        let g = build_chain(&["A","B","C","D"], 1);
        assert!(compute_treewidth_upper_bound(&g) <= 1);
    }

    #[test]
    fn test_complete_k4() {
        let mut g = RtigGraph::new();
        let names = ["A","B","C","D"];
        for n in &names { g.add_service(sid(n)); }
        let p = ResiliencePolicy::empty();
        for i in 0..4 { for j in 0..4 { if i != j { g.add_dependency(&sid(names[i]), &sid(names[j]), p.clone()); } } }
        assert_eq!(compute_treewidth_upper_bound(&g), 3);
    }

    #[test]
    fn test_tree_treewidth() {
        // Star: center -> A,B,C,D. Tree has treewidth 1.
        let mut g = RtigGraph::new();
        for n in &["center","A","B","C","D"] { g.add_service(sid(n)); }
        let p = ResiliencePolicy::empty();
        for n in &["A","B","C","D"] { g.add_dependency(&sid("center"), &sid(n), p.clone()); }
        assert!(compute_treewidth_upper_bound(&g) <= 2);
    }

    #[test]
    fn test_diamond() {
        let g = build_diamond(1);
        let tw = compute_treewidth_upper_bound(&g);
        assert!(tw <= 3);
    }

    #[test]
    fn test_min_fill_vs_min_degree() {
        let g = build_chain(&["A","B","C","D","E"], 1);
        let md = compute_treewidth_upper_bound(&g);
        let mf = compute_treewidth_upper_bound_min_fill(&g);
        assert!(md <= 2);
        assert!(mf <= 2);
    }

    #[test]
    fn test_greedy_decomposition_valid() {
        let g = build_chain(&["A","B","C"], 1);
        let td = greedy_tree_decomposition(&g);
        assert!(td.validate(&g));
    }

    #[test]
    fn test_greedy_decomposition_diamond() {
        let g = build_diamond(1);
        let td = greedy_tree_decomposition(&g);
        assert!(td.validate(&g));
        assert!(td.width() <= 3);
    }

    #[test]
    fn test_estimator_dispatch() {
        let g = build_chain(&["A","B","C"], 1);
        let est = TreewidthEstimator::new();
        let md = est.estimate(&g, EstimationMethod::MinDegree);
        let mf = est.estimate(&g, EstimationMethod::MinFill);
        assert!(md <= 2);
        assert!(mf <= 2);
    }

    #[test]
    fn test_estimate_all() {
        let g = build_diamond(1);
        let est = TreewidthEstimator::new();
        let all = est.estimate_all_methods(&g);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_nice_decomposition() {
        let g = build_chain(&["A","B","C"], 1);
        let td = greedy_tree_decomposition(&g);
        let nice = to_nice_decomposition(&td);
        assert!(nice.num_bags() >= td.num_bags());
    }

    #[test]
    fn test_bag_operations() {
        let b1 = Bag::new(vec![sid("A"), sid("B"), sid("C")]);
        let b2 = Bag::new(vec![sid("B"), sid("C"), sid("D")]);
        assert!(b1.contains(&sid("A")));
        assert!(!b1.contains(&sid("D")));
        let inter = b1.intersection(&b2);
        assert_eq!(inter.len(), 2);
    }

    #[test]
    fn test_large_graph() {
        let names: Vec<String> = (0..15).map(|i| format!("S{}", i)).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let g = build_chain(&name_refs, 1);
        let tw = compute_treewidth_upper_bound(&g);
        assert!(tw <= 2);
    }
}
