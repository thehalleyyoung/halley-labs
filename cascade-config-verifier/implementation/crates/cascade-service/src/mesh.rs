//! Service mesh modelling — core types and builder for constructing
//! dependency graphs of micro-services with resilience policies.

use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode as GraphServiceNode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The type of dependency between two services.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    Synchronous,
    Asynchronous,
    EventDriven,
}

impl Default for DependencyType {
    fn default() -> Self {
        Self::Synchronous
    }
}

/// A single service in the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshService {
    pub id: String,
    pub name: String,
    pub namespace: String,
    pub tier: usize,
    pub capacity: u64,
    pub baseline_load: u64,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

fn default_timeout() -> u64 {
    30_000
}

/// A directed dependency edge between two services.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDependency {
    pub source: String,
    pub target: String,
    pub retry_count: u32,
    pub timeout_ms: u64,
    pub dependency_type: DependencyType,
}

/// Per-service resilience configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResilienceConfig {
    pub retry_count: u32,
    pub timeout_ms: u64,
    pub per_try_timeout_ms: u64,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            retry_count: 3,
            timeout_ms: 3000,
            per_try_timeout_ms: 1000,
        }
    }
}

/// Describes the blast-radius of a service failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlastRadius {
    pub directly_affected: Vec<String>,
    pub transitively_affected: Vec<String>,
    pub impact_score: f64,
}

// ---------------------------------------------------------------------------
// ServiceMesh
// ---------------------------------------------------------------------------

/// The complete service mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMesh {
    pub services: Vec<MeshService>,
    pub dependencies: Vec<MeshDependency>,
    pub policies: HashMap<String, ResilienceConfig>,
}

impl ServiceMesh {
    /// Look up a service by id.
    pub fn get_service(&self, id: &str) -> Option<&MeshService> {
        self.services.iter().find(|s| s.id == id)
    }

    /// All dependencies originating from the given service.
    pub fn get_dependencies_from(&self, id: &str) -> Vec<&MeshDependency> {
        self.dependencies.iter().filter(|d| d.source == id).collect()
    }

    /// All dependencies targeting the given service.
    pub fn get_dependencies_to(&self, id: &str) -> Vec<&MeshDependency> {
        self.dependencies.iter().filter(|d| d.target == id).collect()
    }

    /// Group services by tier, returning sorted `(tier, services)`.
    pub fn get_service_tiers(&self) -> Vec<(usize, Vec<&MeshService>)> {
        let mut tiers: HashMap<usize, Vec<&MeshService>> = HashMap::new();
        for svc in &self.services {
            tiers.entry(svc.tier).or_default().push(svc);
        }
        let mut result: Vec<(usize, Vec<&MeshService>)> = tiers.into_iter().collect();
        result.sort_by_key(|(t, _)| *t);
        result
    }

    /// Compute the blast-radius of `failed` going down.
    pub fn compute_blast_radius(&self, failed: &str) -> BlastRadius {
        let directly_affected: Vec<String> = self
            .dependencies
            .iter()
            .filter(|d| d.target == failed)
            .map(|d| d.source.clone())
            .collect();

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(failed.to_string());
        for da in &directly_affected {
            visited.insert(da.clone());
        }
        let mut queue: VecDeque<String> = directly_affected.iter().cloned().collect();
        let mut transitive: Vec<String> = Vec::new();

        while let Some(current) = queue.pop_front() {
            for dep in &self.dependencies {
                if dep.target == current && !visited.contains(&dep.source) {
                    visited.insert(dep.source.clone());
                    transitive.push(dep.source.clone());
                    queue.push_back(dep.source.clone());
                }
            }
        }

        let total_services = self.services.len().max(1) as f64;
        let impact_score =
            (directly_affected.len() as f64 + transitive.len() as f64 * 0.5) / total_services;

        BlastRadius {
            directly_affected,
            transitively_affected: transitive,
            impact_score,
        }
    }

    /// Estimate steady-state load distribution given `entry_load` injected at
    /// root services (those with no incoming dependencies).
    pub fn estimate_load_distribution(&self, entry_load: u64) -> HashMap<String, u64> {
        let mut loads: HashMap<String, u64> = HashMap::new();
        let targets: HashSet<&str> = self.dependencies.iter().map(|d| d.target.as_str()).collect();

        let roots: Vec<&str> = self
            .services
            .iter()
            .map(|s| s.id.as_str())
            .filter(|id| !targets.contains(id))
            .collect();

        if roots.is_empty() {
            let per = entry_load / self.services.len().max(1) as u64;
            for svc in &self.services {
                loads.insert(svc.id.clone(), per.max(svc.baseline_load));
            }
            return loads;
        }

        let per_root = entry_load / roots.len().max(1) as u64;
        for r in &roots {
            loads.insert(r.to_string(), per_root);
        }

        let order = self.topological_order();
        for svc_id in &order {
            let incoming: Vec<&MeshDependency> = self
                .dependencies
                .iter()
                .filter(|d| d.target == *svc_id)
                .collect();
            if !incoming.is_empty() {
                let mut total: u64 = 0;
                for dep in &incoming {
                    let src_load = loads.get(&dep.source).copied().unwrap_or(0);
                    let factor = 1 + dep.retry_count as u64;
                    total = total.saturating_add(src_load.saturating_mul(factor));
                }
                let baseline = self
                    .get_service(svc_id)
                    .map(|s| s.baseline_load)
                    .unwrap_or(0);
                let current = loads.get(svc_id.as_str()).copied().unwrap_or(0);
                loads.insert(svc_id.clone(), current.max(total).max(baseline));
            }
        }

        for svc in &self.services {
            loads
                .entry(svc.id.clone())
                .and_modify(|v| *v = (*v).max(svc.baseline_load))
                .or_insert(svc.baseline_load);
        }

        loads
    }

    /// Convert to adjacency list.
    pub fn to_adjacency_list(&self) -> Vec<(String, String, u32, u64, u64)> {
        self.dependencies
            .iter()
            .map(|d| {
                let target_baseline = self
                    .get_service(&d.target)
                    .map(|s| s.baseline_load)
                    .unwrap_or(0);
                (
                    d.source.clone(),
                    d.target.clone(),
                    d.retry_count,
                    d.timeout_ms,
                    target_baseline,
                )
            })
            .collect()
    }

    /// Validate the mesh; returns an empty vec when valid.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let ids: HashSet<&str> = self.services.iter().map(|s| s.id.as_str()).collect();

        let mut seen_ids: HashSet<&str> = HashSet::new();
        for svc in &self.services {
            if !seen_ids.insert(&svc.id) {
                problems.push(format!("Duplicate service id: {}", svc.id));
            }
        }

        for dep in &self.dependencies {
            if !ids.contains(dep.source.as_str()) {
                problems.push(format!(
                    "Dependency source '{}' not found in services",
                    dep.source
                ));
            }
            if !ids.contains(dep.target.as_str()) {
                problems.push(format!(
                    "Dependency target '{}' not found in services",
                    dep.target
                ));
            }
            if dep.source == dep.target {
                problems.push(format!("Self-dependency on '{}'", dep.source));
            }
        }

        for svc in &self.services {
            if svc.capacity == 0 {
                problems.push(format!("Service '{}' has zero capacity", svc.id));
            }
            if svc.baseline_load > svc.capacity {
                problems.push(format!(
                    "Service '{}' baseline_load ({}) exceeds capacity ({})",
                    svc.id, svc.baseline_load, svc.capacity
                ));
            }
        }

        for (svc_id, _) in &self.policies {
            if !ids.contains(svc_id.as_str()) {
                problems.push(format!("Policy references unknown service '{}'", svc_id));
            }
        }

        let mut edge_set: HashSet<(&str, &str)> = HashSet::new();
        for dep in &self.dependencies {
            if !edge_set.insert((&dep.source, &dep.target)) {
                problems.push(format!(
                    "Duplicate dependency from '{}' to '{}'",
                    dep.source, dep.target
                ));
            }
        }

        problems
    }

    /// Kahn's algorithm topological sort (best-effort for cyclic graphs).
    pub(crate) fn topological_order(&self) -> Vec<String> {
        let ids: Vec<&str> = self.services.iter().map(|s| s.id.as_str()).collect();
        let mut in_degree: HashMap<&str, usize> = ids.iter().map(|id| (*id, 0usize)).collect();
        for dep in &self.dependencies {
            if let Some(d) = in_degree.get_mut(dep.target.as_str()) {
                *d += 1;
            }
        }
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut order: Vec<String> = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for dep in &self.dependencies {
                if dep.source == node {
                    if let Some(d) = in_degree.get_mut(dep.target.as_str()) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            queue.push_back(dep.target.as_str());
                        }
                    }
                }
            }
        }
        let order_set: HashSet<String> = order.iter().cloned().collect();
        for svc in &self.services {
            if !order_set.contains(&svc.id) {
                order.push(svc.id.clone());
            }
        }
        order
    }

    /// All service ids.
    pub fn service_ids(&self) -> Vec<&str> {
        self.services.iter().map(|s| s.id.as_str()).collect()
    }

    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    pub fn edge_count(&self) -> usize {
        self.dependencies.len()
    }

    /// Root services (no incoming edges).
    pub fn roots(&self) -> Vec<&MeshService> {
        let targets: HashSet<&str> = self.dependencies.iter().map(|d| d.target.as_str()).collect();
        self.services
            .iter()
            .filter(|s| !targets.contains(s.id.as_str()))
            .collect()
    }

    /// Leaf services (no outgoing edges).
    pub fn leaves(&self) -> Vec<&MeshService> {
        let sources: HashSet<&str> = self.dependencies.iter().map(|d| d.source.as_str()).collect();
        self.services
            .iter()
            .filter(|s| !sources.contains(s.id.as_str()))
            .collect()
    }

    /// Build an [`RtigGraph`] from this mesh for graph-based analyses.
    pub fn graph(&self) -> RtigGraph {
        let mut g = RtigGraph::new();
        for svc in &self.services {
            let node = GraphServiceNode::new(&svc.id, svc.capacity as u32)
                .with_baseline_load(svc.baseline_load as u32)
                .with_tier(svc.tier as u32)
                .with_timeout_ms(svc.timeout_ms);
            g.add_service_node(&node);
        }
        for dep in &self.dependencies {
            let edge = DependencyEdgeInfo::new(&dep.source, &dep.target)
                .with_retry_count(dep.retry_count)
                .with_timeout_ms(dep.timeout_ms)
                .with_dep_type(Self::convert_dep_type(&dep.dependency_type));
            g.add_edge(edge);
        }
        g
    }

    fn convert_dep_type(dt: &DependencyType) -> cascade_types::topology::DependencyType {
        match dt {
            DependencyType::Synchronous => cascade_types::topology::DependencyType::Synchronous,
            DependencyType::Asynchronous => cascade_types::topology::DependencyType::Asynchronous,
            DependencyType::EventDriven => cascade_types::topology::DependencyType::EventDriven,
        }
    }
}

// ---------------------------------------------------------------------------
// Stub types for mesh analysis (used by higher-level modules)
// ---------------------------------------------------------------------------

/// A critical path through the service mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub services: Vec<String>,
    pub total_latency_ms: u64,
}

/// Health model for the mesh.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshHealthModel {
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unhealthy_count: usize,
}

/// Topology summary for the mesh.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshTopology {
    pub layers: usize,
    pub max_depth: usize,
}

/// Mesh validator for checking correctness.
#[derive(Debug, Clone, Default)]
pub struct MeshValidator;

/// Service criticality tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ServiceTier {
    Standard = 0,
    Important = 1,
    Critical = 2,
}

/// Traffic model for load estimation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrafficModel {
    pub rps: f64,
}

/// Edge-level resilience policies.
#[derive(Debug, Clone, Default)]
pub struct MeshPolicies {
    pub edge_policies: HashMap<(String, String), EdgePolicy>,
}

/// Per-edge resilience policy.
#[derive(Debug, Clone, Default)]
pub struct EdgePolicy {
    pub retry: Option<cascade_types::policy::RetryPolicy>,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
}

/// Minimal circuit breaker config stub.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub threshold: u32,
}

// ---------------------------------------------------------------------------
// MeshBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct MeshBuilder {
    services: Vec<MeshService>,
    dependencies: Vec<MeshDependency>,
    policies: HashMap<String, ResilienceConfig>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_service(
        &mut self,
        id: &str,
        name: &str,
        namespace: &str,
        tier: usize,
        capacity: u64,
        baseline_load: u64,
    ) -> &mut Self {
        self.services.push(MeshService {
            id: id.to_string(),
            name: name.to_string(),
            namespace: namespace.to_string(),
            tier,
            capacity,
            baseline_load,
            timeout_ms: 30_000,
        });
        self
    }

    /// Add a service from a [`GraphServiceNode`], for fluent builder chaining.
    pub fn add_node(mut self, node: GraphServiceNode) -> Result<Self, String> {
        self.services.push(MeshService {
            id: node.id.clone(),
            name: node.id.clone(),
            namespace: "default".to_string(),
            tier: node.tier as usize,
            capacity: node.capacity as u64,
            baseline_load: node.baseline_load as u64,
            timeout_ms: node.timeout_ms,
        });
        Ok(self)
    }

    pub fn add_dependency(
        &mut self,
        source: &str,
        target: &str,
        retry_count: u32,
        timeout_ms: u64,
    ) -> &mut Self {
        self.dependencies.push(MeshDependency {
            source: source.to_string(),
            target: target.to_string(),
            retry_count,
            timeout_ms,
            dependency_type: DependencyType::Synchronous,
        });
        self
    }

    pub fn add_dependency_typed(
        &mut self,
        source: &str,
        target: &str,
        retry_count: u32,
        timeout_ms: u64,
        dep_type: DependencyType,
    ) -> &mut Self {
        self.dependencies.push(MeshDependency {
            source: source.to_string(),
            target: target.to_string(),
            retry_count,
            timeout_ms,
            dependency_type: dep_type,
        });
        self
    }

    pub fn set_policy(&mut self, service_id: &str, config: ResilienceConfig) -> &mut Self {
        self.policies.insert(service_id.to_string(), config);
        self
    }

    /// Add a dependency with retries (fluent, consuming builder).
    pub fn add_dependency_with_retries(
        mut self,
        source: &str,
        target: &str,
        dep_type: cascade_types::topology::DependencyType,
        retries: u32,
        timeout_ms: u64,
    ) -> Result<Self, String> {
        self.dependencies.push(MeshDependency {
            source: source.to_string(),
            target: target.to_string(),
            retry_count: retries,
            timeout_ms,
            dependency_type: Self::from_types_dep_type(dep_type),
        });
        Ok(self)
    }

    /// Add a dependency with default retries (fluent, consuming builder).
    pub fn add_dep(
        mut self,
        source: &str,
        target: &str,
        dep_type: cascade_types::topology::DependencyType,
    ) -> Result<Self, String> {
        self.dependencies.push(MeshDependency {
            source: source.to_string(),
            target: target.to_string(),
            retry_count: 0,
            timeout_ms: 30_000,
            dependency_type: Self::from_types_dep_type(dep_type),
        });
        Ok(self)
    }

    fn from_types_dep_type(dt: cascade_types::topology::DependencyType) -> DependencyType {
        match dt {
            cascade_types::topology::DependencyType::Synchronous => DependencyType::Synchronous,
            cascade_types::topology::DependencyType::Asynchronous => DependencyType::Asynchronous,
            cascade_types::topology::DependencyType::EventDriven => DependencyType::EventDriven,
        }
    }

    pub fn build(self) -> Result<ServiceMesh, String> {
        if self.services.is_empty() {
            return Err("Mesh must contain at least one service".to_string());
        }
        let mesh = ServiceMesh {
            services: self.services,
            dependencies: self.dependencies,
            policies: self.policies,
        };
        let problems = mesh.validate();
        if problems.is_empty() {
            Ok(mesh)
        } else {
            Err(problems.join("; "))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_mesh() -> ServiceMesh {
        let mut b = MeshBuilder::new();
        b.add_service("gateway", "Gateway", "default", 0, 10000, 500);
        b.add_service("auth", "Auth", "default", 1, 5000, 200);
        b.add_service("users", "Users", "default", 2, 3000, 100);
        b.add_dependency("gateway", "auth", 2, 1000);
        b.add_dependency("auth", "users", 1, 500);
        b.build().unwrap()
    }

    #[test]
    fn test_builder_creates_mesh() {
        let mesh = simple_mesh();
        assert_eq!(mesh.services.len(), 3);
        assert_eq!(mesh.dependencies.len(), 2);
    }

    #[test]
    fn test_builder_empty_fails() {
        let b = MeshBuilder::new();
        assert!(b.build().is_err());
    }

    #[test]
    fn test_get_service() {
        let mesh = simple_mesh();
        assert_eq!(mesh.get_service("auth").unwrap().name, "Auth");
        assert!(mesh.get_service("nonexistent").is_none());
    }

    #[test]
    fn test_dependencies_from() {
        let mesh = simple_mesh();
        let deps = mesh.get_dependencies_from("gateway");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].target, "auth");
    }

    #[test]
    fn test_dependencies_to() {
        let mesh = simple_mesh();
        let deps = mesh.get_dependencies_to("auth");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].source, "gateway");
    }

    #[test]
    fn test_service_tiers() {
        let mesh = simple_mesh();
        let tiers = mesh.get_service_tiers();
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0].0, 0);
        assert_eq!(tiers[0].1[0].id, "gateway");
    }

    #[test]
    fn test_blast_radius_middle() {
        let mesh = simple_mesh();
        let br = mesh.compute_blast_radius("auth");
        assert!(br.directly_affected.contains(&"gateway".to_string()));
        assert!(br.impact_score > 0.0);
    }

    #[test]
    fn test_blast_radius_leaf() {
        let mesh = simple_mesh();
        let br = mesh.compute_blast_radius("users");
        assert!(br.directly_affected.contains(&"auth".to_string()));
        assert!(br.transitively_affected.contains(&"gateway".to_string()));
    }

    #[test]
    fn test_blast_radius_root() {
        let mesh = simple_mesh();
        let br = mesh.compute_blast_radius("gateway");
        assert!(br.directly_affected.is_empty());
        assert_eq!(br.impact_score, 0.0);
    }

    #[test]
    fn test_load_distribution() {
        let mesh = simple_mesh();
        let loads = mesh.estimate_load_distribution(1000);
        assert!(*loads.get("gateway").unwrap() >= 1000);
        assert!(*loads.get("auth").unwrap() >= 3000);
    }

    #[test]
    fn test_adjacency_list() {
        let mesh = simple_mesh();
        let adj = mesh.to_adjacency_list();
        assert_eq!(adj.len(), 2);
    }

    #[test]
    fn test_validate_clean() {
        let mesh = simple_mesh();
        assert!(mesh.validate().is_empty());
    }

    #[test]
    fn test_validate_bad_reference() {
        let mesh = ServiceMesh {
            services: vec![MeshService {
                id: "a".into(), name: "A".into(), namespace: "ns".into(),
                tier: 0, capacity: 100, baseline_load: 10,
            }],
            dependencies: vec![MeshDependency {
                source: "a".into(), target: "missing".into(),
                retry_count: 0, timeout_ms: 100,
                dependency_type: DependencyType::Synchronous,
            }],
            policies: HashMap::new(),
        };
        assert!(mesh.validate().iter().any(|p| p.contains("not found")));
    }

    #[test]
    fn test_validate_self_dependency() {
        let mesh = ServiceMesh {
            services: vec![MeshService {
                id: "a".into(), name: "A".into(), namespace: "ns".into(),
                tier: 0, capacity: 100, baseline_load: 10,
            }],
            dependencies: vec![MeshDependency {
                source: "a".into(), target: "a".into(),
                retry_count: 0, timeout_ms: 100,
                dependency_type: DependencyType::Synchronous,
            }],
            policies: HashMap::new(),
        };
        assert!(mesh.validate().iter().any(|p| p.contains("Self-dependency")));
    }

    #[test]
    fn test_validate_zero_capacity() {
        let mesh = ServiceMesh {
            services: vec![MeshService {
                id: "a".into(), name: "A".into(), namespace: "ns".into(),
                tier: 0, capacity: 0, baseline_load: 0,
            }],
            dependencies: vec![],
            policies: HashMap::new(),
        };
        assert!(mesh.validate().iter().any(|p| p.contains("zero capacity")));
    }

    #[test]
    fn test_dependency_type_default() {
        assert_eq!(DependencyType::default(), DependencyType::Synchronous);
    }

    #[test]
    fn test_builder_set_policy() {
        let mut b = MeshBuilder::new();
        b.add_service("a", "A", "ns", 0, 100, 10);
        b.set_policy("a", ResilienceConfig {
            retry_count: 5, timeout_ms: 2000, per_try_timeout_ms: 400,
        });
        let mesh = b.build().unwrap();
        assert_eq!(mesh.policies.get("a").unwrap().retry_count, 5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mesh = simple_mesh();
        let json = serde_json::to_string(&mesh).unwrap();
        let back: ServiceMesh = serde_json::from_str(&json).unwrap();
        assert_eq!(back.services.len(), mesh.services.len());
        assert_eq!(back.dependencies.len(), mesh.dependencies.len());
    }

    #[test]
    fn test_roots_and_leaves() {
        let mesh = simple_mesh();
        assert_eq!(mesh.roots().len(), 1);
        assert_eq!(mesh.roots()[0].id, "gateway");
        assert_eq!(mesh.leaves().len(), 1);
        assert_eq!(mesh.leaves()[0].id, "users");
    }

    #[test]
    fn test_topological_order() {
        let mesh = simple_mesh();
        let order = mesh.topological_order();
        let gw = order.iter().position(|s| s == "gateway").unwrap();
        let au = order.iter().position(|s| s == "auth").unwrap();
        let us = order.iter().position(|s| s == "users").unwrap();
        assert!(gw < au);
        assert!(au < us);
    }
}
