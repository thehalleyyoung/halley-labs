//! Retry-Timeout Interaction Graph (RTIG) — the core graph data structure.

use cascade_types::policy::ResiliencePolicy;
use cascade_types::service::ServiceId;
use indexmap::IndexMap;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("service `{0}` not found")]
    ServiceNotFound(String),
    #[error("service `{0}` already exists")]
    DuplicateService(String),
    #[error("dangling dependency: source `{src}` or target `{tgt}` not in graph")]
    DanglingEdge { src: String, tgt: String },
    #[error("self-loop on service `{0}`")]
    SelfLoop(String),
    #[error("graph validation failed: {0}")]
    Validation(String),
}

// ---------------------------------------------------------------------------
// GraphStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub service_count: usize,
    pub dependency_count: usize,
    pub max_fan_in: usize,
    pub max_fan_out: usize,
    pub is_dag: bool,
    pub cycle_count: usize,
    pub source_count: usize,
    pub sink_count: usize,
    pub diameter: usize,
    pub max_amplification_factor: u32,
}

// ---------------------------------------------------------------------------
// RtigGraph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RtigGraph {
    pub(crate) inner: DiGraph<ServiceId, ResiliencePolicy>,
    pub(crate) node_map: IndexMap<ServiceId, NodeIndex>,
    pub(crate) node_store: IndexMap<String, ServiceNode>,
}

impl RtigGraph {
    pub fn new() -> Self {
        Self { inner: DiGraph::new(), node_map: IndexMap::new(), node_store: IndexMap::new() }
    }

    pub fn add_service(&mut self, id: ServiceId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) { return idx; }
        let idx = self.inner.add_node(id.clone());
        self.node_map.insert(id, idx);
        idx
    }

    pub fn add_dependency(&mut self, from: &ServiceId, to: &ServiceId, policy: ResiliencePolicy) -> Option<EdgeIndex> {
        let src = *self.node_map.get(from)?;
        let tgt = *self.node_map.get(to)?;
        Some(self.inner.add_edge(src, tgt, policy))
    }

    pub fn remove_service(&mut self, id: &ServiceId) -> bool {
        if let Some(&idx) = self.node_map.get(id) {
            self.inner.remove_node(idx);
            self.node_map.swap_remove(id);
            self.rebuild_node_map();
            true
        } else { false }
    }

    pub fn remove_dependency(&mut self, from: &ServiceId, to: &ServiceId) -> bool {
        let src = match self.node_map.get(from) { Some(&i) => i, None => return false };
        let tgt = match self.node_map.get(to) { Some(&i) => i, None => return false };
        if let Some(e) = self.inner.find_edge(src, tgt) { self.inner.remove_edge(e); true } else { false }
    }

    pub fn service_count(&self) -> usize { self.inner.node_count() }
    pub fn dependency_count(&self) -> usize { self.inner.edge_count() }
    pub fn edge_count(&self) -> usize { self.inner.edge_count() }
    pub fn contains_service(&self, id: &ServiceId) -> bool { self.node_map.contains_key(id) }

    pub fn get_node_index(&self, id: &ServiceId) -> Option<NodeIndex> {
        self.node_map.get(id).copied()
    }

    pub fn services(&self) -> Vec<ServiceId> { self.node_map.keys().cloned().collect() }

    pub fn get_edge_policy(&self, from: &ServiceId, to: &ServiceId) -> Option<&ResiliencePolicy> {
        let src = *self.node_map.get(from)?;
        let tgt = *self.node_map.get(to)?;
        self.inner.edge_weight(self.inner.find_edge(src, tgt)?)
    }

    pub fn get_predecessors(&self, id: &ServiceId) -> Vec<ServiceId> {
        let idx = match self.node_map.get(id) { Some(&i) => i, None => return Vec::new() };
        self.inner.neighbors_directed(idx, Direction::Incoming)
            .filter_map(|n| self.inner.node_weight(n).cloned()).collect()
    }

    pub fn get_successors(&self, id: &ServiceId) -> Vec<ServiceId> {
        let idx = match self.node_map.get(id) { Some(&i) => i, None => return Vec::new() };
        self.inner.neighbors_directed(idx, Direction::Outgoing)
            .filter_map(|n| self.inner.node_weight(n).cloned()).collect()
    }

    // -- Path enumeration --------------------------------------------------

    pub fn get_all_paths(&self, source: &ServiceId, target: &ServiceId) -> Vec<Vec<ServiceId>> {
        self.get_simple_paths(source, target, self.inner.node_count())
    }

    pub fn get_simple_paths(&self, source: &ServiceId, target: &ServiceId, max_length: usize) -> Vec<Vec<ServiceId>> {
        let src = match self.node_map.get(source) { Some(&i) => i, None => return Vec::new() };
        let tgt = match self.node_map.get(target) { Some(&i) => i, None => return Vec::new() };
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(src);
        let mut path = vec![src];
        self.dfs_paths(src, tgt, &mut visited, &mut path, &mut results, max_length);
        results.into_iter().map(|p| p.into_iter()
            .filter_map(|n| self.inner.node_weight(n).cloned()).collect()).collect()
    }

    fn dfs_paths(&self, current: NodeIndex, target: NodeIndex, visited: &mut HashSet<NodeIndex>,
                 path: &mut Vec<NodeIndex>, results: &mut Vec<Vec<NodeIndex>>, max_length: usize) {
        if current == target { results.push(path.clone()); return; }
        if path.len() > max_length { return; }
        for neighbor in self.inner.neighbors_directed(current, Direction::Outgoing) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                path.push(neighbor);
                self.dfs_paths(neighbor, target, visited, path, results, max_length);
                path.pop();
                visited.remove(&neighbor);
            }
        }
    }

    // -- Structural metrics ------------------------------------------------

    pub fn compute_diameter(&self) -> usize {
        let mut max_dist = 0usize;
        for &src in self.node_map.values() {
            for d in self.bfs_distances(src).values() {
                if *d > max_dist { max_dist = *d; }
            }
        }
        max_dist
    }

    pub fn compute_depth(&self, root: &ServiceId) -> usize {
        let idx = match self.node_map.get(root) { Some(&i) => i, None => return 0 };
        self.bfs_distances(idx).values().copied().max().unwrap_or(0)
    }

    fn bfs_distances(&self, start: NodeIndex) -> IndexMap<NodeIndex, usize> {
        let mut dist = IndexMap::new();
        let mut queue = VecDeque::new();
        dist.insert(start, 0usize);
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            let d = dist[&v];
            for n in self.inner.neighbors_directed(v, Direction::Outgoing) {
                if !dist.contains_key(&n) { dist.insert(n, d + 1); queue.push_back(n); }
            }
        }
        dist
    }

    pub fn compute_treewidth_estimate(&self) -> usize {
        if self.inner.node_count() == 0 { return 0; }
        // Min-degree elimination heuristic on underlying undirected graph.
        let mut adj: IndexMap<NodeIndex, HashSet<NodeIndex>> = IndexMap::new();
        for idx in self.inner.node_indices() { adj.entry(idx).or_default(); }
        for e in self.inner.edge_references() {
            adj.entry(e.source()).or_default().insert(e.target());
            adj.entry(e.target()).or_default().insert(e.source());
        }
        let mut max_clique = 0usize;
        while !adj.is_empty() {
            let (&v, _) = adj.iter().min_by_key(|(_, nbrs)| nbrs.len()).unwrap();
            let nbrs: Vec<NodeIndex> = adj[&v].iter().copied().collect();
            if nbrs.len() > max_clique { max_clique = nbrs.len(); }
            for i in 0..nbrs.len() {
                for j in (i + 1)..nbrs.len() {
                    let (a, b) = (nbrs[i], nbrs[j]);
                    if adj.contains_key(&a) && adj.contains_key(&b) {
                        adj.get_mut(&a).unwrap().insert(b);
                        adj.get_mut(&b).unwrap().insert(a);
                    }
                }
            }
            adj.swap_remove(&v);
            for s in adj.values_mut() { s.remove(&v); }
        }
        max_clique
    }

    pub fn get_fan_in_services(&self, threshold: usize) -> Vec<(ServiceId, usize)> {
        let mut r = Vec::new();
        for (id, &idx) in &self.node_map {
            let c = self.inner.neighbors_directed(idx, Direction::Incoming).count();
            if c >= threshold { r.push((id.clone(), c)); }
        }
        r.sort_by(|a, b| b.1.cmp(&a.1));
        r
    }

    pub fn get_fan_out_services(&self, threshold: usize) -> Vec<(ServiceId, usize)> {
        let mut r = Vec::new();
        for (id, &idx) in &self.node_map {
            let c = self.inner.neighbors_directed(idx, Direction::Outgoing).count();
            if c >= threshold { r.push((id.clone(), c)); }
        }
        r.sort_by(|a, b| b.1.cmp(&a.1));
        r
    }

    // -- DAG operations ----------------------------------------------------

    pub fn topological_sort(&self) -> Option<Vec<ServiceId>> {
        let sorted = petgraph::algo::toposort(&self.inner, None).ok()?;
        Some(sorted.into_iter().filter_map(|n| self.inner.node_weight(n).cloned()).collect())
    }

    pub fn is_dag(&self) -> bool { petgraph::algo::toposort(&self.inner, None).is_ok() }

    pub fn detect_cycles(&self) -> Vec<Vec<ServiceId>> {
        petgraph::algo::tarjan_scc(&self.inner).into_iter()
            .filter(|scc| scc.len() > 1 || (scc.len() == 1 && self.inner.find_edge(scc[0], scc[0]).is_some()))
            .map(|scc| scc.into_iter().filter_map(|n| self.inner.node_weight(n).cloned()).collect())
            .collect()
    }

    // -- Subgraph extraction -----------------------------------------------

    pub fn subgraph(&self, service_ids: &[ServiceId]) -> RtigGraph {
        let mut sub = RtigGraph::new();
        let id_set: HashSet<&ServiceId> = service_ids.iter().collect();
        for id in service_ids {
            if self.node_map.contains_key(id) { sub.add_service(id.clone()); }
        }
        for e in self.inner.edge_references() {
            let src = self.inner.node_weight(e.source()).unwrap();
            let tgt = self.inner.node_weight(e.target()).unwrap();
            if id_set.contains(src) && id_set.contains(tgt) {
                sub.add_dependency(src, tgt, e.weight().clone());
            }
        }
        sub
    }

    /// All services that can reach `target` (backward slice).
    pub fn cone_of_influence(&self, target: &ServiceId) -> RtigGraph {
        let idx = match self.node_map.get(target) { Some(&i) => i, None => return RtigGraph::new() };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(idx); queue.push_back(idx);
        while let Some(v) = queue.pop_front() {
            for n in self.inner.neighbors_directed(v, Direction::Incoming) {
                if visited.insert(n) { queue.push_back(n); }
            }
        }
        let ids: Vec<ServiceId> = visited.iter().filter_map(|&n| self.inner.node_weight(n).cloned()).collect();
        self.subgraph(&ids)
    }

    // -- Statistics --------------------------------------------------------

    pub fn graph_stats(&self) -> GraphStats {
        let max_fan_in = self.node_map.values()
            .map(|&i| self.inner.neighbors_directed(i, Direction::Incoming).count()).max().unwrap_or(0);
        let max_fan_out = self.node_map.values()
            .map(|&i| self.inner.neighbors_directed(i, Direction::Outgoing).count()).max().unwrap_or(0);
        let source_count = self.node_map.values()
            .filter(|&&i| self.inner.neighbors_directed(i, Direction::Incoming).count() == 0).count();
        let sink_count = self.node_map.values()
            .filter(|&&i| self.inner.neighbors_directed(i, Direction::Outgoing).count() == 0).count();
        let cycles = self.detect_cycles();
        let max_amp = self.inner.edge_weights().map(|p| p.amplification_factor()).max().unwrap_or(1);
        GraphStats {
            service_count: self.service_count(), dependency_count: self.dependency_count(),
            max_fan_in, max_fan_out, is_dag: cycles.is_empty(), cycle_count: cycles.len(),
            source_count, sink_count, diameter: self.compute_diameter(),
            max_amplification_factor: max_amp,
        }
    }

    // -- Export / serialization ---------------------------------------------

    pub fn to_dot_format(&self) -> String {
        let mut dot = String::from("digraph RTIG {\n  rankdir=LR;\n");
        for (id, _) in &self.node_map {
            dot.push_str(&format!("  \"{}\";\n", id.as_str()));
        }
        for e in self.inner.edge_references() {
            let s = self.inner.node_weight(e.source()).unwrap();
            let t = self.inner.node_weight(e.target()).unwrap();
            let a = e.weight().amplification_factor();
            dot.push_str(&format!("  \"{}\" -> \"{}\" [label=\"amp={}\"];\n", s.as_str(), t.as_str(), a));
        }
        dot.push_str("}\n");
        dot
    }

    pub fn to_adjacency_list(&self) -> Vec<(ServiceId, ServiceId, ResiliencePolicy)> {
        self.inner.edge_references().map(|e| {
            (self.inner.node_weight(e.source()).unwrap().clone(),
             self.inner.node_weight(e.target()).unwrap().clone(),
             e.weight().clone())
        }).collect()
    }

    fn rebuild_node_map(&mut self) {
        self.node_map.clear();
        for idx in self.inner.node_indices() {
            if let Some(id) = self.inner.node_weight(idx) { self.node_map.insert(id.clone(), idx); }
        }
    }
}

// ---------------------------------------------------------------------------
// EdgeInfo — lightweight edge descriptor used by symmetry / analysis modules
// ---------------------------------------------------------------------------

/// A simple edge descriptor exposing source, target, and amplification.
#[derive(Debug, Clone)]
pub struct EdgeInfo {
    pub source: String,
    pub target: String,
    amplification: u32,
}

impl EdgeInfo {
    pub fn amplification_factor(&self) -> u32 {
        self.amplification
    }
}

// -- Convenience helpers removed: see &str-based API below (line ~746) ------

impl Default for RtigGraph { fn default() -> Self { Self::new() } }

impl Serialize for RtigGraph {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let services: Vec<String> = self.node_map.keys().map(|s| s.0.clone()).collect();
        let edges: Vec<(String, String)> = self.inner.edge_references().map(|e| {
            (self.inner.node_weight(e.source()).unwrap().0.clone(),
             self.inner.node_weight(e.target()).unwrap().0.clone())
        }).collect();
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("RtigGraph", 2)?;
        st.serialize_field("services", &services)?;
        st.serialize_field("edges", &edges)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for RtigGraph {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw { services: Vec<String>, edges: Vec<(String, String)> }
        let raw = Raw::deserialize(deserializer)?;
        let mut g = RtigGraph::new();
        for s in &raw.services { g.add_service(ServiceId::new(s.clone())); }
        for (s, t) in &raw.edges {
            g.add_dependency(&ServiceId::new(s.clone()), &ServiceId::new(t.clone()), ResiliencePolicy::empty());
        }
        Ok(g)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct RtigGraphBuilder {
    services: Vec<ServiceId>,
    edges: Vec<(ServiceId, ServiceId, ResiliencePolicy)>,
}

impl RtigGraphBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn add_service(mut self, id: ServiceId) -> Self { self.services.push(id); self }
    pub fn add_dependency(mut self, from: ServiceId, to: ServiceId, policy: ResiliencePolicy) -> Self {
        self.edges.push((from, to, policy)); self
    }
    pub fn build(self) -> Result<RtigGraph, GraphError> {
        let mut g = RtigGraph::new();
        for s in &self.services { g.add_service(s.clone()); }
        for (from, to, policy) in &self.edges {
            if !g.contains_service(from) || !g.contains_service(to) {
                return Err(GraphError::DanglingEdge { src: from.0.clone(), tgt: to.0.clone() });
            }
            if from == to { return Err(GraphError::SelfLoop(from.0.clone())); }
            g.add_dependency(from, to, policy.clone());
        }
        Ok(g)
    }
}

// ---------------------------------------------------------------------------
// ServiceNode — annotated graph vertex
// ---------------------------------------------------------------------------

/// A service node with metadata used by higher-level analyses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceNode {
    pub id: String,
    pub capacity: u32,
    pub baseline_load: u32,
    pub tier: u32,
    pub timeout_ms: u64,
    pub retry_budget: u32,
    pub health: cascade_types::service::ServiceHealth,
}

impl ServiceNode {
    pub fn new(id: &str, capacity: u32) -> Self {
        Self {
            id: id.to_string(),
            capacity,
            baseline_load: 0,
            tier: 0,
            timeout_ms: 30_000,
            retry_budget: 0,
            health: cascade_types::service::ServiceHealth::Healthy,
        }
    }

    pub fn with_baseline_load(mut self, load: u32) -> Self {
        self.baseline_load = load;
        self
    }

    pub fn with_tier(mut self, tier: u32) -> Self {
        self.tier = tier;
        self
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_retry_budget(mut self, budget: u32) -> Self {
        self.retry_budget = budget;
        self
    }

    pub fn with_health(mut self, health: cascade_types::service::ServiceHealth) -> Self {
        self.health = health;
        self
    }

    pub fn headroom(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        1.0 - (self.baseline_load as f64 / self.capacity as f64)
    }

    pub fn is_overloaded(&self) -> bool {
        self.baseline_load > self.capacity
    }
}

impl From<ServiceNode> for ServiceId {
    fn from(node: ServiceNode) -> Self {
        ServiceId::new(node.id)
    }
}

// ---------------------------------------------------------------------------
// DependencyEdgeInfo — annotated graph edge
// ---------------------------------------------------------------------------

/// Edge metadata for dependency links in the RTIG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyEdgeInfo {
    pub source: String,
    pub target: String,
    pub retry_count: u32,
    pub timeout_ms: u64,
    pub weight: f64,
    pub dep_type: cascade_types::topology::DependencyType,
}

impl DependencyEdgeInfo {
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            retry_count: 0,
            timeout_ms: 30_000,
            weight: 1.0,
            dep_type: cascade_types::topology::DependencyType::Synchronous,
        }
    }

    pub fn with_retry_count(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    pub fn with_dep_type(mut self, dep_type: cascade_types::topology::DependencyType) -> Self {
        self.dep_type = dep_type;
        self
    }

    pub fn amplification_factor(&self) -> u32 {
        1 + self.retry_count
    }

    /// Amplification factor as f64 for floating-point calculations.
    pub fn amplification_factor_f64(&self) -> f64 {
        (1 + self.retry_count) as f64
    }

    /// Convert into a [`ResiliencePolicy`] for use with the core graph.
    pub fn to_resilience_policy(&self) -> ResiliencePolicy {
        if self.retry_count > 0 {
            ResiliencePolicy::empty()
                .with_retry(
                    cascade_types::policy::RetryPolicy::new(self.retry_count)
                        .with_per_try_timeout(self.timeout_ms),
                )
                .with_timeout(cascade_types::policy::TimeoutPolicy::new(self.timeout_ms))
        } else {
            ResiliencePolicy::empty()
                .with_timeout(cascade_types::policy::TimeoutPolicy::new(self.timeout_ms))
        }
    }
}

// ---------------------------------------------------------------------------
// RtigGraph — convenience methods for annotated types
// ---------------------------------------------------------------------------

impl RtigGraph {
    /// Add a service from a [`ServiceNode`], converting to [`ServiceId`].
    pub fn add_service_node(&mut self, node: &ServiceNode) -> NodeIndex {
        self.node_store.insert(node.id.clone(), node.clone());
        self.add_service(ServiceId::new(node.id.clone()))
    }

    /// Add an edge described by a [`DependencyEdgeInfo`].
    pub fn add_edge(&mut self, info: DependencyEdgeInfo) -> Option<EdgeIndex> {
        let src = ServiceId::new(info.source.clone());
        let tgt = ServiceId::new(info.target.clone());
        self.add_dependency(&src, &tgt, info.to_resilience_policy())
    }

    /// Return all incoming edges for the service identified by `id_str`.
    pub fn incoming_edges(&self, id_str: &str) -> Vec<DependencyEdgeInfo> {
        let id = ServiceId::new(id_str);
        let idx = match self.node_map.get(&id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.inner
            .edges_directed(idx, Direction::Incoming)
            .map(|e| {
                let src = self.inner.node_weight(e.source()).unwrap();
                let tgt = self.inner.node_weight(e.target()).unwrap();
                let pol = e.weight();
                let retry_count = pol
                    .retry
                    .as_ref()
                    .map_or(0, |r| r.max_retries);
                let timeout_ms = pol
                    .timeout
                    .as_ref()
                    .map_or(30_000, |t| t.request_timeout_ms);
                DependencyEdgeInfo {
                    source: src.0.clone(),
                    target: tgt.0.clone(),
                    retry_count,
                    timeout_ms,
                    weight: 1.0,
                    dep_type: cascade_types::topology::DependencyType::Synchronous,
                }
            })
            .collect()
    }

    /// Return all outgoing edges for the service identified by `id_str`.
    pub fn outgoing_edges(&self, id_str: &str) -> Vec<DependencyEdgeInfo> {
        let id = ServiceId::new(id_str);
        let idx = match self.node_map.get(&id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| {
                let src = self.inner.node_weight(e.source()).unwrap();
                let tgt = self.inner.node_weight(e.target()).unwrap();
                let pol = e.weight();
                let retry_count = pol.retry.as_ref().map_or(0, |r| r.max_retries);
                let timeout_ms = pol
                    .timeout
                    .as_ref()
                    .map_or(30_000, |t| t.request_timeout_ms);
                DependencyEdgeInfo {
                    source: src.0.clone(),
                    target: tgt.0.clone(),
                    retry_count,
                    timeout_ms,
                    weight: 1.0,
                    dep_type: cascade_types::topology::DependencyType::Synchronous,
                }
            })
            .collect()
    }

    /// Total amplification factor along the highest-amplification path between
    /// two services, computed as the product of per-edge factors.
    pub fn path_amplification(&self, from: &str, to: &str) -> u32 {
        let src = ServiceId::new(from);
        let tgt = ServiceId::new(to);
        let paths = self.get_all_paths(&src, &tgt);
        paths
            .iter()
            .map(|path| {
                path.windows(2)
                    .map(|w| {
                        self.get_edge_policy(&w[0], &w[1])
                            .map_or(1, |p| p.amplification_factor())
                    })
                    .product::<u32>()
            })
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// RtigGraph — convenience &str-based API for cascade-service
// ---------------------------------------------------------------------------

impl RtigGraph {
    /// Service IDs as string slices.
    pub fn service_ids(&self) -> Vec<&str> {
        self.node_map.keys().map(|id| id.as_str()).collect()
    }

    /// Predecessors by string ID.
    pub fn predecessors(&self, id: &str) -> Vec<&str> {
        let sid = ServiceId::new(id);
        let idx = match self.node_map.get(&sid) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.inner
            .neighbors_directed(idx, Direction::Incoming)
            .filter_map(|n| self.inner.node_weight(n).map(|s| s.as_str()))
            .collect()
    }

    /// Successors by string ID.
    pub fn successors(&self, id: &str) -> Vec<&str> {
        let sid = ServiceId::new(id);
        let idx = match self.node_map.get(&sid) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.inner
            .neighbors_directed(idx, Direction::Outgoing)
            .filter_map(|n| self.inner.node_weight(n).map(|s| s.as_str()))
            .collect()
    }

    /// Number of incoming dependencies.
    pub fn fan_in(&self, id: &str) -> usize {
        self.predecessors(id).len()
    }

    /// Number of outgoing dependencies.
    pub fn fan_out(&self, id: &str) -> usize {
        self.successors(id).len()
    }

    /// Services with no incoming edges.
    pub fn roots(&self) -> Vec<&str> {
        self.service_ids()
            .into_iter()
            .filter(|id| self.fan_in(id) == 0)
            .collect()
    }

    /// Services with no outgoing edges.
    pub fn leaves(&self) -> Vec<&str> {
        self.service_ids()
            .into_iter()
            .filter(|id| self.fan_out(id) == 0)
            .collect()
    }

    /// BFS forward reachability from `source`.
    pub fn forward_reachable(&self, source: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(source.to_string());
        queue.push_back(source.to_string());
        while let Some(cur) = queue.pop_front() {
            for next in self.successors(&cur) {
                if visited.insert(next.to_string()) {
                    queue.push_back(next.to_string());
                }
            }
        }
        visited
    }

    /// BFS reverse reachability: services that transitively depend on `target`.
    pub fn reverse_reachable(&self, target: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(target.to_string());
        queue.push_back(target.to_string());
        while let Some(cur) = queue.pop_front() {
            for prev in self.predecessors(&cur) {
                if visited.insert(prev.to_string()) {
                    queue.push_back(prev.to_string());
                }
            }
        }
        visited
    }

    /// Graph diameter.
    pub fn diameter(&self) -> usize {
        self.compute_diameter()
    }

    /// All edges as [`DependencyEdgeInfo`].
    pub fn edges(&self) -> Vec<DependencyEdgeInfo> {
        self.inner
            .edge_references()
            .map(|e| {
                let src = self.inner.node_weight(e.source()).unwrap();
                let tgt = self.inner.node_weight(e.target()).unwrap();
                let pol = e.weight();
                let retry_count = pol.retry.as_ref().map_or(0, |r| r.max_retries);
                let timeout_ms = pol.timeout.as_ref().map_or(30_000, |t| t.request_timeout_ms);
                DependencyEdgeInfo {
                    source: src.0.clone(),
                    target: tgt.0.clone(),
                    retry_count,
                    timeout_ms,
                    weight: 1.0,
                    dep_type: cascade_types::topology::DependencyType::Synchronous,
                }
            })
            .collect()
    }

    /// Lookup a stored [`ServiceNode`] by string ID.
    pub fn service(&self, id: &str) -> Option<&ServiceNode> {
        self.node_store.get(id)
    }

    /// Longest path length from any root to `target`.
    pub fn longest_path_to(&self, target: &str) -> usize {
        let sorted = match self.topological_sort() {
            Some(s) => s,
            None => return 0,
        };
        let mut dist: HashMap<String, usize> = HashMap::new();
        for svc in &sorted {
            let d = self.predecessors(svc.as_str())
                .iter()
                .filter_map(|p| dist.get(*p).copied())
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            dist.insert(svc.as_str().to_string(), d);
        }
        dist.get(target).copied().unwrap_or(0)
    }

    /// Maximum retry count on any edge.
    pub fn max_retries(&self) -> u32 {
        self.inner
            .edge_references()
            .map(|e| e.weight().retry.as_ref().map_or(0, |r| r.max_retries))
            .max()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Convenience builder that works with ServiceNode / DependencyEdgeInfo
// ---------------------------------------------------------------------------

/// An ergonomic builder that accepts [`ServiceNode`] and [`DependencyEdgeInfo`]
/// directly (as opposed to [`RtigGraphBuilder`] which works with raw types).
#[derive(Debug, Default)]
pub struct RtigGraphSimpleBuilder {
    nodes: Vec<ServiceNode>,
    edges: Vec<DependencyEdgeInfo>,
}

impl RtigGraphSimpleBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn add_service(mut self, node: ServiceNode) -> Self {
        self.nodes.push(node);
        self
    }

    pub fn add_edge(mut self, edge: DependencyEdgeInfo) -> Self {
        self.edges.push(edge);
        self
    }

    pub fn build(self) -> RtigGraph {
        let mut g = RtigGraph::new();
        for n in &self.nodes {
            g.add_service_node(n);
        }
        for e in self.edges {
            g.add_edge(e);
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Create a simple chain: names[0] -> names[1] -> ... -> names[n-1]
pub fn build_chain(names: &[&str], retries: u32) -> RtigGraph {
    let mut g = RtigGraph::new();
    let ids: Vec<ServiceId> = names.iter().map(|n| ServiceId::new(*n)).collect();
    for id in &ids { g.add_service(id.clone()); }
    for w in ids.windows(2) {
        g.add_dependency(&w[0], &w[1],
            ResiliencePolicy::empty().with_retry(cascade_types::policy::RetryPolicy::new(retries)));
    }
    g
}

/// Create a diamond: A->{B,C}, {B,C}->D
pub fn build_diamond(retries: u32) -> RtigGraph {
    let mut g = RtigGraph::new();
    for n in &["A","B","C","D"] { g.add_service(ServiceId::new(*n)); }
    let p = || ResiliencePolicy::empty().with_retry(cascade_types::policy::RetryPolicy::new(retries));
    let (a,b,c,d) = (ServiceId::new("A"),ServiceId::new("B"),ServiceId::new("C"),ServiceId::new("D"));
    g.add_dependency(&a, &b, p());
    g.add_dependency(&a, &c, p());
    g.add_dependency(&b, &d, p());
    g.add_dependency(&c, &d, p());
    g
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_types::policy::RetryPolicy;

    fn sid(s: &str) -> ServiceId { ServiceId::new(s) }
    fn pr(n: u32) -> ResiliencePolicy { ResiliencePolicy::empty().with_retry(RetryPolicy::new(n)) }

    #[test] fn test_empty_graph() { let g = RtigGraph::new(); assert_eq!(g.service_count(), 0); assert!(g.is_dag()); }
    #[test] fn test_single_node() { let mut g = RtigGraph::new(); g.add_service(sid("A")); assert!(g.contains_service(&sid("A"))); }
    #[test] fn test_add_dup() { let mut g = RtigGraph::new(); let a = g.add_service(sid("A")); let b = g.add_service(sid("A")); assert_eq!(a, b); }
    #[test] fn test_chain() { let g = build_chain(&["A","B","C","D"], 2); assert_eq!(g.service_count(), 4); assert_eq!(g.dependency_count(), 3); assert!(g.is_dag()); }
    #[test] fn test_diamond() { let g = build_diamond(2); assert_eq!(g.get_fan_in_services(2)[0].0, sid("D")); }
    #[test] fn test_all_paths() { let g = build_diamond(1); assert_eq!(g.get_all_paths(&sid("A"), &sid("D")).len(), 2); }
    #[test] fn test_simple_paths_limit() { let g = build_chain(&["A","B","C","D","E"], 1); assert!(g.get_simple_paths(&sid("A"), &sid("E"), 2).is_empty()); assert_eq!(g.get_simple_paths(&sid("A"), &sid("E"), 5).len(), 1); }
    #[test] fn test_topo() { assert_eq!(build_chain(&["A","B","C"], 1).topological_sort().unwrap(), vec![sid("A"),sid("B"),sid("C")]); }
    #[test] fn test_cycle() { let mut g = RtigGraph::new(); for n in &["A","B","C"]{g.add_service(sid(n));} let p = ResiliencePolicy::empty(); g.add_dependency(&sid("A"),&sid("B"),p.clone()); g.add_dependency(&sid("B"),&sid("C"),p.clone()); g.add_dependency(&sid("C"),&sid("A"),p); assert!(!g.is_dag()); assert_eq!(g.detect_cycles().len(), 1); }
    #[test] fn test_fan() { let g = build_diamond(2); assert_eq!(g.get_fan_in_services(2)[0].0, sid("D")); assert_eq!(g.get_fan_out_services(2)[0].0, sid("A")); }
    #[test] fn test_diameter() { assert_eq!(build_chain(&["A","B","C","D"], 1).compute_diameter(), 3); }
    #[test] fn test_depth() { let g = build_chain(&["A","B","C"], 1); assert_eq!(g.compute_depth(&sid("A")), 2); }
    #[test] fn test_subgraph() { let s = build_chain(&["A","B","C","D"], 1).subgraph(&[sid("B"),sid("C")]); assert_eq!(s.service_count(), 2); assert_eq!(s.dependency_count(), 1); }
    #[test] fn test_cone() { let c = build_chain(&["A","B","C","D"], 1).cone_of_influence(&sid("C")); assert_eq!(c.service_count(), 3); assert!(!c.contains_service(&sid("D"))); }
    #[test] fn test_rm_svc() { let mut g = build_chain(&["A","B","C"], 1); g.remove_service(&sid("B")); assert_eq!(g.service_count(), 2); assert_eq!(g.dependency_count(), 0); }
    #[test] fn test_rm_dep() { let mut g = build_chain(&["A","B","C"], 1); g.remove_dependency(&sid("A"),&sid("B")); assert_eq!(g.dependency_count(), 1); }
    #[test] fn test_edge_pol() { let mut g = RtigGraph::new(); g.add_service(sid("A")); g.add_service(sid("B")); g.add_dependency(&sid("A"), &sid("B"), pr(5)); assert_eq!(g.get_edge_policy(&sid("A"), &sid("B")).unwrap().amplification_factor(), 6); }
    #[test] fn test_dot() { let d = build_chain(&["A","B"], 2).to_dot_format(); assert!(d.contains("RTIG")); }
    #[test] fn test_stats() { let s = build_diamond(3).graph_stats(); assert_eq!(s.service_count, 4); assert!(s.is_dag); assert_eq!(s.max_fan_in, 2); }
    #[test] fn test_builder_ok() { let g = RtigGraphBuilder::new().add_service(sid("A")).add_service(sid("B")).add_dependency(sid("A"),sid("B"),ResiliencePolicy::empty()).build().unwrap(); assert_eq!(g.service_count(), 2); }
    #[test] fn test_builder_dangle() { assert!(RtigGraphBuilder::new().add_service(sid("A")).add_dependency(sid("A"),sid("B"),ResiliencePolicy::empty()).build().is_err()); }
    #[test] fn test_builder_loop() { assert!(RtigGraphBuilder::new().add_service(sid("A")).add_dependency(sid("A"),sid("A"),ResiliencePolicy::empty()).build().is_err()); }
    #[test] fn test_serde() { let g = build_chain(&["X","Y","Z"], 1); let g2: RtigGraph = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap(); assert_eq!(g2.service_count(), 3); }
    #[test] fn test_tw() { assert!(build_chain(&["A","B","C","D"], 1).compute_treewidth_estimate() <= 2); }
}
