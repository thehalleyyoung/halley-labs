//! Topology construction from service meshes and configuration sources.

use crate::rtig::RtigGraph;
use cascade_types::config::{EnvoyConfig, IstioConfig, KubernetesConfig};
use cascade_types::policy::{ResiliencePolicy, RetryPolicy, TimeoutPolicy};
use cascade_types::service::ServiceId;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A frozen snapshot of a built topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySnapshot {
    pub graph: RtigGraph,
    pub version: String,
    pub source: String,
}

/// Warning produced during topology construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyWarning {
    pub message: String,
    pub severity: WarningSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// Difference between two topology snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDiff {
    pub added_services: Vec<ServiceId>,
    pub removed_services: Vec<ServiceId>,
    pub added_edges: Vec<(ServiceId, ServiceId)>,
    pub removed_edges: Vec<(ServiceId, ServiceId)>,
}

impl TopologyDiff {
    pub fn is_empty(&self) -> bool {
        self.added_services.is_empty()
            && self.removed_services.is_empty()
            && self.added_edges.is_empty()
            && self.removed_edges.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ServiceResolver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ServiceResolver {
    services: IndexMap<ServiceId, ResiliencePolicy>,
}

impl ServiceResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, id: ServiceId, default_policy: ResiliencePolicy) {
        self.services.insert(id, default_policy);
    }

    pub fn resolve(&self, id: &ServiceId) -> Option<&ResiliencePolicy> {
        self.services.get(id)
    }

    pub fn contains(&self, id: &ServiceId) -> bool {
        self.services.contains_key(id)
    }

    pub fn all_ids(&self) -> Vec<ServiceId> {
        self.services.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// PolicyResolver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct PolicyResolver {
    overrides: HashMap<(ServiceId, ServiceId), ResiliencePolicy>,
}

impl PolicyResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_override(&mut self, from: &ServiceId, to: &ServiceId, policy: ResiliencePolicy) {
        self.overrides.insert((from.clone(), to.clone()), policy);
    }

    pub fn resolve(
        &self,
        from: &ServiceId,
        to: &ServiceId,
        default: &ResiliencePolicy,
    ) -> ResiliencePolicy {
        self.overrides
            .get(&(from.clone(), to.clone()))
            .cloned()
            .unwrap_or_else(|| default.clone())
    }

    pub fn has_override(&self, from: &ServiceId, to: &ServiceId) -> bool {
        self.overrides.contains_key(&(from.clone(), to.clone()))
    }
}

// ---------------------------------------------------------------------------
// TopologyBuilder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct TopologyBuilder {
    services: ServiceResolver,
    policies: PolicyResolver,
    edges: Vec<(ServiceId, ServiceId)>,
    warnings: Vec<TopologyWarning>,
}

impl TopologyBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_service(mut self, id: ServiceId, policy: ResiliencePolicy) -> Self {
        self.services.register(id, policy);
        self
    }

    pub fn add_dependency(mut self, from: ServiceId, to: ServiceId) -> Self {
        self.edges.push((from, to));
        self
    }

    pub fn with_policy_resolver(mut self, pr: PolicyResolver) -> Self {
        self.policies = pr;
        self
    }

    /// Build the graph from Kubernetes-style configs.
    pub fn from_kubernetes_configs(configs: &[KubernetesConfig]) -> Self {
        let mut builder = Self::new();
        for config in configs {
            let name = config.metadata.qualified_name();
            let id = ServiceId::new(&name);
            builder = builder.add_service(id, ResiliencePolicy::empty());
        }
        builder
    }

    /// Build from Istio virtual service configs.
    pub fn from_istio_configs(configs: &[IstioConfig]) -> Self {
        let mut builder = Self::new();
        for config in configs {
            for (i, vs) in config.virtual_services.iter().enumerate() {
                let id = ServiceId::new(&vs.name);
                builder = builder.add_service(id.clone(), ResiliencePolicy::empty());
                if i > 0 {
                    let prev = ServiceId::new(&config.virtual_services[i - 1].name);
                    builder = builder.add_dependency(prev, id);
                }
            }
        }
        builder
    }

    /// Build from Envoy cluster configs.
    pub fn from_envoy_configs(configs: &[EnvoyConfig]) -> Self {
        let mut builder = Self::new();
        for config in configs {
            for cluster in &config.clusters {
                let id = ServiceId::new(&cluster.name);
                builder = builder.add_service(id, ResiliencePolicy::empty());
            }
        }
        builder
    }

    /// Merge another builder into this one.
    pub fn merge(mut self, other: TopologyBuilder) -> Self {
        for id in other.services.all_ids() {
            if let Some(p) = other.services.resolve(&id) {
                if !self.services.contains(&id) {
                    self.services.register(id, p.clone());
                } else {
                    self.warnings.push(TopologyWarning {
                        message: format!("Duplicate service {} in merge", id),
                        severity: WarningSeverity::Low,
                    });
                }
            }
        }
        for (from, to) in other.edges {
            self.edges.push((from, to));
        }
        self
    }

    /// Validate the topology and return warnings.
    pub fn validate(&self) -> Vec<TopologyWarning> {
        let mut warns = self.warnings.clone();
        for (from, to) in &self.edges {
            if !self.services.contains(from) {
                warns.push(TopologyWarning {
                    message: format!("Dangling edge source: {}", from),
                    severity: WarningSeverity::High,
                });
            }
            if !self.services.contains(to) {
                warns.push(TopologyWarning {
                    message: format!("Dangling edge target: {}", to),
                    severity: WarningSeverity::High,
                });
            }
            if from == to {
                warns.push(TopologyWarning {
                    message: format!("Self-loop on {}", from),
                    severity: WarningSeverity::Medium,
                });
            }
        }
        warns
    }

    /// Infer default retry/timeout policies where none are explicitly set.
    pub fn infer_defaults(mut self, retry: u32, timeout_ms: u64) -> Self {
        let default_policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(retry))
            .with_timeout(TimeoutPolicy::new(timeout_ms));
        for id in self.services.all_ids() {
            if self
                .services
                .resolve(&id)
                .map(|p| !p.has_any())
                .unwrap_or(true)
            {
                self.services.register(id, default_policy.clone());
            }
        }
        self
    }

    /// Build the final RtigGraph.
    pub fn build(self) -> TopologySnapshot {
        let mut graph = RtigGraph::new();
        for id in self.services.all_ids() {
            graph.add_service(id);
        }
        for (from, to) in &self.edges {
            let default = self
                .services
                .resolve(from)
                .cloned()
                .unwrap_or_else(ResiliencePolicy::empty);
            let policy = self.policies.resolve(from, to, &default);
            graph.add_dependency(from, to, policy);
        }
        TopologySnapshot {
            graph,
            version: "1.0".to_string(),
            source: "builder".to_string(),
        }
    }

    /// Diff two topology snapshots.
    pub fn diff(old: &TopologySnapshot, new: &TopologySnapshot) -> TopologyDiff {
        let old_svcs: std::collections::HashSet<_> = old.graph.services().into_iter().collect();
        let new_svcs: std::collections::HashSet<_> = new.graph.services().into_iter().collect();
        let old_edges: std::collections::HashSet<(ServiceId, ServiceId)> =
            old.graph.to_adjacency_list().into_iter().map(|(s, t, _)| (s, t)).collect();
        let new_edges: std::collections::HashSet<(ServiceId, ServiceId)> =
            new.graph.to_adjacency_list().into_iter().map(|(s, t, _)| (s, t)).collect();

        TopologyDiff {
            added_services: new_svcs.difference(&old_svcs).cloned().collect(),
            removed_services: old_svcs.difference(&new_svcs).cloned().collect(),
            added_edges: new_edges.difference(&old_edges).cloned().collect(),
            removed_edges: old_edges.difference(&new_edges).cloned().collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_types::policy::{ResiliencePolicy, RetryPolicy};
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    #[test]
    fn test_build_simple() {
        let snap = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_service(sid("B"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("B"))
            .build();
        assert_eq!(snap.graph.service_count(), 2);
        assert_eq!(snap.graph.dependency_count(), 1);
    }

    #[test]
    fn test_policy_override() {
        let mut pr = PolicyResolver::new();
        let override_policy = ResiliencePolicy::empty().with_retry(RetryPolicy::new(5));
        pr.set_override(&sid("A"), &sid("B"), override_policy.clone());
        assert!(pr.has_override(&sid("A"), &sid("B")));
        let resolved = pr.resolve(&sid("A"), &sid("B"), &ResiliencePolicy::empty());
        assert_eq!(resolved.amplification_factor(), 6); // 1+5
    }

    #[test]
    fn test_service_resolver() {
        let mut sr = ServiceResolver::new();
        sr.register(sid("svc"), ResiliencePolicy::empty());
        assert!(sr.contains(&sid("svc")));
        assert!(!sr.contains(&sid("missing")));
    }

    #[test]
    fn test_merge() {
        let b1 = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty());
        let b2 = TopologyBuilder::new()
            .add_service(sid("B"), ResiliencePolicy::empty());
        let snap = b1.merge(b2).add_dependency(sid("A"), sid("B")).build();
        assert_eq!(snap.graph.service_count(), 2);
        assert_eq!(snap.graph.dependency_count(), 1);
    }

    #[test]
    fn test_validate_dangling() {
        let builder = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("Z")); // Z doesn't exist
        let warns = builder.validate();
        assert!(!warns.is_empty());
    }

    #[test]
    fn test_diff() {
        let old = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_service(sid("B"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("B"))
            .build();
        let new = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_service(sid("C"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("C"))
            .build();
        let diff = TopologyBuilder::diff(&old, &new);
        assert!(!diff.is_empty());
        assert!(diff.added_services.contains(&sid("C")));
        assert!(diff.removed_services.contains(&sid("B")));
    }

    #[test]
    fn test_infer_defaults() {
        let snap = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_service(sid("B"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("B"))
            .infer_defaults(3, 5000)
            .build();
        assert_eq!(snap.graph.service_count(), 2);
    }

    #[test]
    fn test_topology_warning_severity() {
        let w = TopologyWarning {
            message: "test".into(),
            severity: WarningSeverity::High,
        };
        assert_eq!(w.severity, WarningSeverity::High);
    }

    #[test]
    fn test_topology_diff_empty() {
        let snap = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .build();
        let diff = TopologyBuilder::diff(&snap, &snap);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_validate_self_loop() {
        let builder = TopologyBuilder::new()
            .add_service(sid("A"), ResiliencePolicy::empty())
            .add_dependency(sid("A"), sid("A"));
        let warns = builder.validate();
        assert!(warns.iter().any(|w| w.severity == WarningSeverity::Medium));
    }

    #[test]
    fn test_policy_resolver_no_override() {
        let pr = PolicyResolver::new();
        let default = ResiliencePolicy::empty();
        let resolved = pr.resolve(&sid("A"), &sid("B"), &default);
        assert_eq!(resolved.amplification_factor(), 1);
    }
}
