use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::policy::ResiliencePolicy;
use crate::service::ServiceId;

// ── EdgeId ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub String);

impl EdgeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn from_endpoints(source: &ServiceId, target: &ServiceId) -> Self {
        Self(format!("{}->{}", source.as_str(), target.as_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for EdgeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for EdgeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// ── DependencyType ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    Synchronous,
    Asynchronous,
    EventDriven,
}

impl DependencyType {
    pub fn propagates_failure(&self) -> bool {
        matches!(self, DependencyType::Synchronous)
    }

    pub fn propagates_latency(&self) -> bool {
        matches!(self, DependencyType::Synchronous)
    }

    pub fn is_fire_and_forget(&self) -> bool {
        matches!(
            self,
            DependencyType::Asynchronous | DependencyType::EventDriven
        )
    }
}

impl Default for DependencyType {
    fn default() -> Self {
        DependencyType::Synchronous
    }
}

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyType::Synchronous => write!(f, "sync"),
            DependencyType::Asynchronous => write!(f, "async"),
            DependencyType::EventDriven => write!(f, "event-driven"),
        }
    }
}

// ── EdgeWeight ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeWeight {
    pub amplification_factor: f64,
    pub timeout_budget: f64,
    pub reliability: f64,
}

impl EdgeWeight {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_policy(policy: &ResiliencePolicy) -> Self {
        let amplification_factor = policy.amplification_factor() as f64;
        let timeout_budget = policy
            .timeout
            .as_ref()
            .map(|t| t.request_timeout_ms as f64)
            .unwrap_or(30000.0);
        Self {
            amplification_factor,
            timeout_budget,
            reliability: 0.99,
        }
    }

    pub fn combined_amplification(&self) -> f64 {
        self.amplification_factor
    }

    pub fn combined_reliability(&self) -> f64 {
        self.reliability
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.amplification_factor < 0.0 {
            errors.push("amplification_factor must be non-negative".to_string());
        }
        if self.timeout_budget < 0.0 {
            errors.push("timeout_budget must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&self.reliability) {
            errors.push("reliability must be between 0.0 and 1.0".to_string());
        }
        errors
    }
}

impl Default for EdgeWeight {
    fn default() -> Self {
        Self {
            amplification_factor: 1.0,
            timeout_budget: 30000.0,
            reliability: 0.99,
        }
    }
}

impl fmt::Display for EdgeWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "amp={:.2} timeout={:.0}ms rel={:.4}",
            self.amplification_factor, self.timeout_budget, self.reliability
        )
    }
}

// ── TopologyMetadata ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyMetadata {
    pub name: String,
    pub description: Option<String>,
    pub source: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub labels: BTreeMap<String, String>,
}

impl TopologyMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            source: None,
            created_at: None,
            labels: BTreeMap::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_created_at(mut self, created_at: chrono::DateTime<chrono::Utc>) -> Self {
        self.created_at = Some(created_at);
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for TopologyMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(ref desc) = self.description {
            write!(f, " ({})", desc)?;
        }
        Ok(())
    }
}

// ── DependencyEdge ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub id: EdgeId,
    pub source: ServiceId,
    pub target: ServiceId,
    pub dependency_type: DependencyType,
    pub policy: ResiliencePolicy,
    pub weight: EdgeWeight,
    pub metadata: BTreeMap<String, String>,
}

impl DependencyEdge {
    pub fn new(source: ServiceId, target: ServiceId) -> Self {
        let id = EdgeId::from_endpoints(&source, &target);
        Self {
            id,
            source,
            target,
            dependency_type: DependencyType::default(),
            policy: ResiliencePolicy::empty(),
            weight: EdgeWeight::default(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_dependency_type(mut self, dep_type: DependencyType) -> Self {
        self.dependency_type = dep_type;
        self
    }

    pub fn with_policy(mut self, policy: ResiliencePolicy) -> Self {
        self.weight = EdgeWeight::from_policy(&policy);
        self.policy = policy;
        self
    }

    pub fn with_weight(mut self, weight: EdgeWeight) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.source == self.target {
            errors.push(format!(
                "self-loop detected: {} -> {}",
                self.source, self.target
            ));
        }
        errors.extend(self.weight.validate());
        errors.extend(self.policy.validate());
        errors
    }
}

impl fmt::Display for DependencyEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {} [{}]",
            self.source, self.target, self.dependency_type
        )
    }
}

// ── PathInfo ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathInfo {
    pub edges: Vec<EdgeId>,
    pub services: Vec<ServiceId>,
    pub amplification_factor: f64,
    pub timeout_budget: f64,
    pub reliability: f64,
}

impl PathInfo {
    pub fn empty(start: ServiceId) -> Self {
        Self {
            edges: Vec::new(),
            services: vec![start],
            amplification_factor: 1.0,
            timeout_budget: f64::INFINITY,
            reliability: 1.0,
        }
    }

    pub fn extend(&mut self, edge: &DependencyEdge) {
        self.edges.push(edge.id.clone());
        self.services.push(edge.target.clone());
        self.amplification_factor *= edge.weight.combined_amplification();
        self.reliability *= edge.weight.combined_reliability();
        self.timeout_budget = self.timeout_budget.min(edge.weight.timeout_budget);
    }

    pub fn length(&self) -> usize {
        self.edges.len()
    }

    pub fn source(&self) -> Option<&ServiceId> {
        self.services.first()
    }

    pub fn target(&self) -> Option<&ServiceId> {
        self.services.last()
    }

    pub fn contains_service(&self, service: &ServiceId) -> bool {
        self.services.contains(service)
    }
}

impl fmt::Display for PathInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path: Vec<&str> = self.services.iter().map(|s| s.as_str()).collect();
        write!(
            f,
            "{} (amp={:.2}, rel={:.4}, timeout={:.0}ms)",
            path.join(" -> "),
            self.amplification_factor,
            self.reliability,
            self.timeout_budget
        )
    }
}

// ── FanInInfo / FanOutInfo ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FanInInfo {
    pub service: ServiceId,
    pub sources: Vec<ServiceId>,
    pub total_amplification: f64,
}

impl FanInInfo {
    pub fn degree(&self) -> usize {
        self.sources.len()
    }
}

impl fmt::Display for FanInInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: fan-in={} amp={:.2}",
            self.service,
            self.degree(),
            self.total_amplification
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FanOutInfo {
    pub service: ServiceId,
    pub targets: Vec<ServiceId>,
    pub max_amplification: f64,
}

impl FanOutInfo {
    pub fn degree(&self) -> usize {
        self.targets.len()
    }
}

impl fmt::Display for FanOutInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: fan-out={} max_amp={:.2}",
            self.service,
            self.degree(),
            self.max_amplification
        )
    }
}

// ── TopologyStats ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyStats {
    pub service_count: usize,
    pub edge_count: usize,
    pub max_depth: usize,
    pub max_fan_in: usize,
    pub max_fan_out: usize,
    pub diameter: usize,
    pub treewidth_estimate: usize,
}

impl TopologyStats {
    pub fn empty() -> Self {
        Self {
            service_count: 0,
            edge_count: 0,
            max_depth: 0,
            max_fan_in: 0,
            max_fan_out: 0,
            diameter: 0,
            treewidth_estimate: 0,
        }
    }
}

impl fmt::Display for TopologyStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "services={} edges={} depth={} fan_in={} fan_out={} diameter={}",
            self.service_count,
            self.edge_count,
            self.max_depth,
            self.max_fan_in,
            self.max_fan_out,
            self.diameter
        )
    }
}

// ── ServiceTopology ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceTopology {
    pub metadata: TopologyMetadata,
    pub services: BTreeSet<ServiceId>,
    outgoing: BTreeMap<ServiceId, BTreeMap<ServiceId, DependencyEdge>>,
    incoming: BTreeMap<ServiceId, BTreeSet<ServiceId>>,
}

impl ServiceTopology {
    pub fn new(metadata: TopologyMetadata) -> Self {
        Self {
            metadata,
            services: BTreeSet::new(),
            outgoing: BTreeMap::new(),
            incoming: BTreeMap::new(),
        }
    }

    // ── Service management ──

    pub fn add_service(&mut self, id: ServiceId) -> bool {
        self.services.insert(id)
    }

    pub fn remove_service(&mut self, id: &ServiceId) -> bool {
        if !self.services.remove(id) {
            return false;
        }
        if let Some(targets) = self.outgoing.remove(id) {
            for target in targets.keys() {
                if let Some(inc) = self.incoming.get_mut(target) {
                    inc.remove(id);
                }
            }
        }
        if let Some(sources) = self.incoming.remove(id) {
            for source in &sources {
                if let Some(out) = self.outgoing.get_mut(source) {
                    out.remove(id);
                }
            }
        }
        true
    }

    pub fn has_service(&self, id: &ServiceId) -> bool {
        self.services.contains(id)
    }

    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    pub fn services(&self) -> impl Iterator<Item = &ServiceId> {
        self.services.iter()
    }

    // ── Edge management ──

    pub fn add_edge(&mut self, edge: DependencyEdge) {
        self.services.insert(edge.source.clone());
        self.services.insert(edge.target.clone());
        self.incoming
            .entry(edge.target.clone())
            .or_default()
            .insert(edge.source.clone());
        self.outgoing
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.target.clone(), edge);
    }

    pub fn remove_edge(
        &mut self,
        source: &ServiceId,
        target: &ServiceId,
    ) -> Option<DependencyEdge> {
        let edge = self
            .outgoing
            .get_mut(source)
            .and_then(|targets| targets.remove(target));
        if edge.is_some() {
            if let Some(inc) = self.incoming.get_mut(target) {
                inc.remove(source);
            }
        }
        edge
    }

    pub fn edge_count(&self) -> usize {
        self.outgoing.values().map(|m| m.len()).sum()
    }

    pub fn edges(&self) -> impl Iterator<Item = &DependencyEdge> {
        self.outgoing.values().flat_map(|m| m.values())
    }

    pub fn get_edge(&self, source: &ServiceId, target: &ServiceId) -> Option<&DependencyEdge> {
        self.outgoing.get(source).and_then(|m| m.get(target))
    }

    pub fn outgoing_edges(&self, id: &ServiceId) -> Vec<&DependencyEdge> {
        self.outgoing
            .get(id)
            .map(|m| m.values().collect())
            .unwrap_or_default()
    }

    pub fn incoming_edges(&self, id: &ServiceId) -> Vec<&DependencyEdge> {
        let sources = match self.incoming.get(id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        sources
            .iter()
            .filter_map(|src| self.outgoing.get(src).and_then(|m| m.get(id)))
            .collect()
    }

    // ── Graph queries ──

    pub fn successors(&self, id: &ServiceId) -> BTreeSet<ServiceId> {
        self.outgoing
            .get(id)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn predecessors(&self, id: &ServiceId) -> BTreeSet<ServiceId> {
        self.incoming.get(id).cloned().unwrap_or_default()
    }

    pub fn roots(&self) -> BTreeSet<ServiceId> {
        self.services
            .iter()
            .filter(|s| {
                self.incoming
                    .get(*s)
                    .map(|inc| inc.is_empty())
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    pub fn leaves(&self) -> BTreeSet<ServiceId> {
        self.services
            .iter()
            .filter(|s| {
                self.outgoing
                    .get(*s)
                    .map(|out| out.is_empty())
                    .unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    // ── Traversals ──

    pub fn reachable_from(&self, start: &ServiceId) -> BTreeSet<ServiceId> {
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());
        visited.insert(start.clone());

        while let Some(current) = queue.pop_front() {
            for successor in self.successors(&current) {
                if visited.insert(successor.clone()) {
                    queue.push_back(successor);
                }
            }
        }
        visited
    }

    pub fn all_paths(&self, source: &ServiceId, target: &ServiceId) -> Vec<PathInfo> {
        let mut results = Vec::new();
        let mut visited = BTreeSet::new();
        visited.insert(source.clone());
        let path = PathInfo::empty(source.clone());
        self.dfs_paths(source, target, &mut visited, path, &mut results);
        results
    }

    fn dfs_paths(
        &self,
        current: &ServiceId,
        target: &ServiceId,
        visited: &mut BTreeSet<ServiceId>,
        path: PathInfo,
        results: &mut Vec<PathInfo>,
    ) {
        if current == target {
            results.push(path);
            return;
        }

        if let Some(neighbors) = self.outgoing.get(current) {
            for (next, edge) in neighbors {
                if !visited.contains(next) {
                    visited.insert(next.clone());
                    let mut extended = path.clone();
                    extended.extend(edge);
                    self.dfs_paths(next, target, visited, extended, results);
                    visited.remove(next);
                }
            }
        }
    }

    pub fn shortest_distances(&self, start: &ServiceId) -> HashMap<ServiceId, usize> {
        let mut distances = HashMap::new();
        distances.insert(start.clone(), 0);
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());

        while let Some(current) = queue.pop_front() {
            let dist = distances[&current];
            for successor in self.successors(&current) {
                if !distances.contains_key(&successor) {
                    distances.insert(successor.clone(), dist + 1);
                    queue.push_back(successor);
                }
            }
        }
        distances
    }

    // ── Fan analysis ──

    pub fn fan_in_info(&self, id: &ServiceId) -> FanInInfo {
        let sources: Vec<ServiceId> = self
            .incoming
            .get(id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        let total_amplification: f64 = sources
            .iter()
            .filter_map(|src| self.get_edge(src, id))
            .map(|e| e.weight.combined_amplification())
            .sum();
        FanInInfo {
            service: id.clone(),
            sources,
            total_amplification,
        }
    }

    pub fn fan_out_info(&self, id: &ServiceId) -> FanOutInfo {
        let targets: Vec<ServiceId> = self
            .outgoing
            .get(id)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        let max_amplification: f64 = self
            .outgoing
            .get(id)
            .map(|m| {
                m.values()
                    .map(|e| e.weight.combined_amplification())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(0.0);
        FanOutInfo {
            service: id.clone(),
            targets,
            max_amplification,
        }
    }

    // ── Topological sort (Kahn's algorithm) ──

    /// Returns None if the graph contains a cycle.
    pub fn topological_sort(&self) -> Option<Vec<ServiceId>> {
        let mut in_degree: HashMap<ServiceId, usize> = HashMap::new();
        for svc in &self.services {
            in_degree.insert(svc.clone(), 0);
        }
        for edge in self.edges() {
            *in_degree.entry(edge.target.clone()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<ServiceId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut sorted = Vec::new();
        while let Some(node) = queue.pop_front() {
            sorted.push(node.clone());
            for successor in self.successors(&node) {
                if let Some(deg) = in_degree.get_mut(&successor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(successor);
                    }
                }
            }
        }

        if sorted.len() == self.services.len() {
            Some(sorted)
        } else {
            None
        }
    }

    // ── Stats ──

    pub fn stats(&self) -> TopologyStats {
        let service_count = self.service_count();
        let edge_count = self.edge_count();
        if service_count == 0 {
            return TopologyStats::empty();
        }

        let max_fan_in = self
            .services
            .iter()
            .map(|s| self.incoming.get(s).map(|inc| inc.len()).unwrap_or(0))
            .max()
            .unwrap_or(0);

        let max_fan_out = self
            .services
            .iter()
            .map(|s| self.outgoing.get(s).map(|out| out.len()).unwrap_or(0))
            .max()
            .unwrap_or(0);

        let mut diameter = 0usize;
        let mut max_depth = 0usize;

        for root in &self.roots() {
            let distances = self.shortest_distances(root);
            if let Some(&d) = distances.values().max() {
                max_depth = max_depth.max(d);
            }
        }

        for svc in &self.services {
            let distances = self.shortest_distances(svc);
            if let Some(&d) = distances.values().max() {
                diameter = diameter.max(d);
            }
        }

        let treewidth_estimate = max_fan_in.min(max_fan_out).saturating_add(1).max(1);

        TopologyStats {
            service_count,
            edge_count,
            max_depth,
            max_fan_in,
            max_fan_out,
            diameter,
            treewidth_estimate,
        }
    }
}

impl fmt::Display for ServiceTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Topology({}: {} services, {} edges)",
            self.metadata.name,
            self.service_count(),
            self.edge_count()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::RetryPolicy;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    fn meta(name: &str) -> TopologyMetadata {
        TopologyMetadata::new(name)
    }

    fn simple_edge(src: &str, tgt: &str) -> DependencyEdge {
        DependencyEdge::new(sid(src), sid(tgt))
    }

    fn build_diamond() -> ServiceTopology {
        // A -> B, A -> C, B -> D, C -> D
        let mut topo = ServiceTopology::new(meta("diamond"));
        topo.add_edge(simple_edge("A", "B"));
        topo.add_edge(simple_edge("A", "C"));
        topo.add_edge(simple_edge("B", "D"));
        topo.add_edge(simple_edge("C", "D"));
        topo
    }

    #[test]
    fn test_edge_id_construction() {
        let eid = EdgeId::new("my-edge");
        assert_eq!(eid.as_str(), "my-edge");
        assert_eq!(eid.to_string(), "my-edge");

        let eid2 = EdgeId::from_endpoints(&sid("svc-a"), &sid("svc-b"));
        assert_eq!(eid2.as_str(), "svc-a->svc-b");
    }

    #[test]
    fn test_edge_id_from_conversions() {
        let from_str: EdgeId = EdgeId::from("hello");
        assert_eq!(from_str.as_str(), "hello");

        let from_string: EdgeId = EdgeId::from(String::from("world"));
        assert_eq!(from_string.as_str(), "world");
    }

    #[test]
    fn test_dependency_type_properties() {
        assert!(DependencyType::Synchronous.propagates_failure());
        assert!(DependencyType::Synchronous.propagates_latency());
        assert!(!DependencyType::Synchronous.is_fire_and_forget());

        assert!(!DependencyType::Asynchronous.propagates_failure());
        assert!(!DependencyType::Asynchronous.propagates_latency());
        assert!(DependencyType::Asynchronous.is_fire_and_forget());

        assert!(!DependencyType::EventDriven.propagates_failure());
        assert!(DependencyType::EventDriven.is_fire_and_forget());
    }

    #[test]
    fn test_dependency_type_default() {
        assert_eq!(DependencyType::default(), DependencyType::Synchronous);
    }

    #[test]
    fn test_edge_weight_default_and_new() {
        let w = EdgeWeight::new();
        assert!((w.amplification_factor - 1.0).abs() < f64::EPSILON);
        assert!((w.timeout_budget - 30000.0).abs() < f64::EPSILON);
        assert!((w.reliability - 0.99).abs() < f64::EPSILON);
        assert!(w.validate().is_empty());
    }

    #[test]
    fn test_edge_weight_from_policy() {
        let policy = ResiliencePolicy::empty().with_retry(RetryPolicy::new(3));
        let w = EdgeWeight::from_policy(&policy);
        assert!((w.amplification_factor - 4.0).abs() < f64::EPSILON);
        assert!((w.timeout_budget - 30000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_weight_validate_invalid() {
        let w = EdgeWeight {
            amplification_factor: -1.0,
            timeout_budget: -5.0,
            reliability: 1.5,
        };
        let errors = w.validate();
        assert_eq!(errors.len(), 3);
    }

    #[test]
    fn test_topology_add_and_count() {
        let mut topo = ServiceTopology::new(meta("test"));
        assert!(topo.add_service(sid("a")));
        assert!(topo.add_service(sid("b")));
        assert!(!topo.add_service(sid("a"))); // duplicate
        assert_eq!(topo.service_count(), 2);
        assert!(topo.has_service(&sid("a")));
        assert!(!topo.has_service(&sid("z")));
        assert_eq!(topo.edge_count(), 0);
    }

    #[test]
    fn test_topology_add_edge_auto_adds_services() {
        let mut topo = ServiceTopology::new(meta("test"));
        topo.add_edge(simple_edge("x", "y"));
        assert!(topo.has_service(&sid("x")));
        assert!(topo.has_service(&sid("y")));
        assert_eq!(topo.service_count(), 2);
        assert_eq!(topo.edge_count(), 1);
    }

    #[test]
    fn test_roots_and_leaves() {
        let topo = build_diamond();
        let roots = topo.roots();
        assert_eq!(roots.len(), 1);
        assert!(roots.contains(&sid("A")));

        let leaves = topo.leaves();
        assert_eq!(leaves.len(), 1);
        assert!(leaves.contains(&sid("D")));
    }

    #[test]
    fn test_successors_predecessors() {
        let topo = build_diamond();
        let succ = topo.successors(&sid("A"));
        assert!(succ.contains(&sid("B")));
        assert!(succ.contains(&sid("C")));
        assert_eq!(succ.len(), 2);

        let pred = topo.predecessors(&sid("D"));
        assert!(pred.contains(&sid("B")));
        assert!(pred.contains(&sid("C")));
        assert_eq!(pred.len(), 2);
    }

    #[test]
    fn test_reachable_from() {
        let topo = build_diamond();
        let reachable = topo.reachable_from(&sid("A"));
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&sid("A")));
        assert!(reachable.contains(&sid("D")));

        let from_b = topo.reachable_from(&sid("B"));
        assert_eq!(from_b.len(), 2); // B, D
        assert!(from_b.contains(&sid("B")));
        assert!(from_b.contains(&sid("D")));
    }

    #[test]
    fn test_all_paths() {
        let topo = build_diamond();
        let paths = topo.all_paths(&sid("A"), &sid("D"));
        assert_eq!(paths.len(), 2);
        for p in &paths {
            assert_eq!(p.source(), Some(&sid("A")));
            assert_eq!(p.target(), Some(&sid("D")));
            assert_eq!(p.length(), 2);
        }
    }

    #[test]
    fn test_shortest_distances() {
        let topo = build_diamond();
        let dists = topo.shortest_distances(&sid("A"));
        assert_eq!(dists[&sid("A")], 0);
        assert_eq!(dists[&sid("B")], 1);
        assert_eq!(dists[&sid("C")], 1);
        assert_eq!(dists[&sid("D")], 2);
    }

    #[test]
    fn test_topological_sort_dag() {
        let topo = build_diamond();
        let sorted = topo.topological_sort();
        assert!(sorted.is_some());
        let sorted = sorted.unwrap();
        assert_eq!(sorted.len(), 4);

        let pos = |s: &str| sorted.iter().position(|x| *x == sid(s)).unwrap();
        assert!(pos("A") < pos("B"));
        assert!(pos("A") < pos("C"));
        assert!(pos("B") < pos("D"));
        assert!(pos("C") < pos("D"));
    }

    #[test]
    fn test_topological_sort_cycle() {
        let mut topo = ServiceTopology::new(meta("cycle"));
        topo.add_edge(simple_edge("X", "Y"));
        topo.add_edge(simple_edge("Y", "Z"));
        topo.add_edge(simple_edge("Z", "X"));
        assert!(topo.topological_sort().is_none());
    }

    #[test]
    fn test_fan_in_fan_out() {
        let topo = build_diamond();

        let fi = topo.fan_in_info(&sid("D"));
        assert_eq!(fi.degree(), 2);
        assert!((fi.total_amplification - 2.0).abs() < f64::EPSILON);

        let fo = topo.fan_out_info(&sid("A"));
        assert_eq!(fo.degree(), 2);
        assert!((fo.max_amplification - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats() {
        let topo = build_diamond();
        let stats = topo.stats();
        assert_eq!(stats.service_count, 4);
        assert_eq!(stats.edge_count, 4);
        assert_eq!(stats.max_depth, 2);
        assert_eq!(stats.max_fan_in, 2);
        assert_eq!(stats.max_fan_out, 2);
        assert_eq!(stats.diameter, 2);
        assert!(stats.treewidth_estimate >= 1);
    }

    #[test]
    fn test_remove_service() {
        let mut topo = build_diamond();
        assert!(topo.remove_service(&sid("B")));
        assert!(!topo.has_service(&sid("B")));
        assert_eq!(topo.service_count(), 3);
        // Edge A->B and B->D should be removed
        assert!(topo.get_edge(&sid("A"), &sid("B")).is_none());
        assert!(topo.get_edge(&sid("B"), &sid("D")).is_none());
        // A->C and C->D should remain
        assert!(topo.get_edge(&sid("A"), &sid("C")).is_some());
        assert!(topo.get_edge(&sid("C"), &sid("D")).is_some());
    }

    #[test]
    fn test_remove_edge() {
        let mut topo = build_diamond();
        let removed = topo.remove_edge(&sid("A"), &sid("B"));
        assert!(removed.is_some());
        assert_eq!(topo.edge_count(), 3);
        assert!(topo.get_edge(&sid("A"), &sid("B")).is_none());
        // Services remain
        assert!(topo.has_service(&sid("A")));
        assert!(topo.has_service(&sid("B")));
    }

    #[test]
    fn test_path_info_extend() {
        let edge1 = DependencyEdge::new(sid("A"), sid("B")).with_weight(EdgeWeight {
            amplification_factor: 2.0,
            timeout_budget: 5000.0,
            reliability: 0.95,
        });
        let edge2 = DependencyEdge::new(sid("B"), sid("C")).with_weight(EdgeWeight {
            amplification_factor: 3.0,
            timeout_budget: 3000.0,
            reliability: 0.90,
        });

        let mut path = PathInfo::empty(sid("A"));
        path.extend(&edge1);
        path.extend(&edge2);

        assert_eq!(path.length(), 2);
        assert_eq!(path.services.len(), 3);
        assert!((path.amplification_factor - 6.0).abs() < f64::EPSILON);
        assert!((path.reliability - 0.855).abs() < 1e-9);
        assert!((path.timeout_budget - 3000.0).abs() < f64::EPSILON);
        assert!(path.contains_service(&sid("B")));
    }

    #[test]
    fn test_edge_validation_self_loop() {
        let edge = DependencyEdge::new(sid("X"), sid("X"));
        let errors = edge.validate();
        assert!(errors.iter().any(|e| e.contains("self-loop")));
    }

    #[test]
    fn test_display_impls() {
        let eid = EdgeId::new("e1");
        assert_eq!(format!("{}", eid), "e1");

        let dt = DependencyType::Asynchronous;
        assert_eq!(format!("{}", dt), "async");

        let w = EdgeWeight::new();
        let display = format!("{}", w);
        assert!(display.contains("amp="));

        let edge = simple_edge("A", "B");
        let display = format!("{}", edge);
        assert!(display.contains("A") && display.contains("B"));

        let topo = build_diamond();
        let display = format!("{}", topo);
        assert!(display.contains("diamond"));
    }

    #[test]
    fn test_empty_stats() {
        let stats = TopologyStats::empty();
        assert_eq!(stats.service_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.max_depth, 0);
        assert_eq!(stats.diameter, 0);
    }

    #[test]
    fn test_topology_metadata_builder() {
        let m = TopologyMetadata::new("test-topo")
            .with_description("A test topology")
            .with_source("unit-test")
            .with_label("env", "test");
        assert_eq!(m.name, "test-topo");
        assert_eq!(m.description.as_deref(), Some("A test topology"));
        assert_eq!(m.source.as_deref(), Some("unit-test"));
        assert_eq!(m.labels.get("env").map(|s| s.as_str()), Some("test"));
        let display = format!("{}", m);
        assert!(display.contains("test-topo"));
        assert!(display.contains("A test topology"));
    }

    #[test]
    fn test_incoming_outgoing_edges() {
        let topo = build_diamond();
        let out = topo.outgoing_edges(&sid("A"));
        assert_eq!(out.len(), 2);
        let inc = topo.incoming_edges(&sid("D"));
        assert_eq!(inc.len(), 2);
    }

    #[test]
    fn test_remove_nonexistent_service() {
        let mut topo = ServiceTopology::new(meta("test"));
        assert!(!topo.remove_service(&sid("ghost")));
    }

    #[test]
    fn test_edge_weight_combined() {
        let w = EdgeWeight {
            amplification_factor: 5.0,
            timeout_budget: 1000.0,
            reliability: 0.8,
        };
        assert!((w.combined_amplification() - 5.0).abs() < f64::EPSILON);
        assert!((w.combined_reliability() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dependency_edge_builder() {
        let edge = DependencyEdge::new(sid("src"), sid("dst"))
            .with_dependency_type(DependencyType::EventDriven)
            .with_metadata("version", "v2");
        assert_eq!(edge.dependency_type, DependencyType::EventDriven);
        assert_eq!(
            edge.metadata.get("version").map(|s| s.as_str()),
            Some("v2")
        );
        assert_eq!(edge.id.as_str(), "src->dst");
    }

    #[test]
    fn test_services_iterator() {
        let mut topo = ServiceTopology::new(meta("test"));
        topo.add_service(sid("z"));
        topo.add_service(sid("a"));
        topo.add_service(sid("m"));
        let names: Vec<&str> = topo.services().map(|s| s.as_str()).collect();
        assert_eq!(names, vec!["a", "m", "z"]); // BTreeSet is sorted
    }

    #[test]
    fn test_stats_display() {
        let stats = TopologyStats::empty();
        let display = format!("{}", stats);
        assert!(display.contains("services=0"));
        assert!(display.contains("edges=0"));
    }
}
