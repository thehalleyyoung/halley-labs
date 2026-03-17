//! Cross-resource reference resolution and dependency inference.
//!
//! This module resolves references between Kubernetes services, Istio
//! VirtualServices / DestinationRules, and their backing Deployments.
//! It builds a dependency graph, detects unresolved references, and infers
//! sensible default policies for services that lack explicit configuration.

use std::collections::{HashSet, VecDeque};

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use log;

use crate::istio::{DestinationRule, IstioConfig, VirtualService};
use crate::kubernetes::{Deployment, KubeService, KubernetesResource};
use crate::{RetryPolicy, ServiceEndpoint, ServiceId, TimeoutPolicy};

// ---------------------------------------------------------------------------
// Core output types
// ---------------------------------------------------------------------------

/// Complete result of a reference-resolution pass.
#[derive(Debug, Clone, Default)]
pub struct ResolvedRefs {
    /// Map from `ServiceId` to the resolved service metadata.
    pub service_map: IndexMap<ServiceId, ResolvedService>,
    /// Map from `ServiceId` to concrete endpoints derived from its ports.
    pub endpoint_map: IndexMap<ServiceId, Vec<ServiceEndpoint>>,
    /// Map from `ServiceId` to resolved retry/timeout policies.
    pub policy_map: IndexMap<ServiceId, ResolvedPolicies>,
    /// Directed dependency edges: (caller, callee).
    pub dependency_graph: Vec<(ServiceId, ServiceId)>,
    /// References that could not be resolved to a concrete service.
    pub unresolved: Vec<UnresolvedRef>,
}

/// Metadata about a single resolved service.
#[derive(Debug, Clone)]
pub struct ResolvedService {
    pub service: ServiceId,
    /// Name of the backing Kubernetes `Service` (if found).
    pub kubernetes_service: Option<String>,
    /// Names of `Deployment` resources that match this service's selector.
    pub deployments: Vec<String>,
    /// Names of Istio `VirtualService` resources that target this service.
    pub virtual_services: Vec<String>,
    /// Names of Istio `DestinationRule` resources that target this service.
    pub destination_rules: Vec<String>,
}

/// Resolved retry & timeout policies for a service, plus their origin.
#[derive(Debug, Clone)]
pub struct ResolvedPolicies {
    pub retry_policy: Option<RetryPolicy>,
    pub timeout_policy: Option<TimeoutPolicy>,
    pub source: PolicySource,
}

/// Where a policy was sourced from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicySource {
    Istio,
    Envoy,
    Annotation,
    Default,
    Merged,
}

/// A reference that could not be resolved.
#[derive(Debug, Clone)]
pub struct UnresolvedRef {
    pub source_kind: String,
    pub source_name: String,
    pub target_host: String,
    pub reason: String,
}

/// Subset-based routing derived from a `DestinationRule` matched against
/// deployment pod-template labels.
#[derive(Debug, Clone, Default)]
pub struct SubsetRouting {
    /// Subsets that matched at least one deployment, with their endpoints.
    pub subsets: IndexMap<String, Vec<ServiceEndpoint>>,
    /// Subset names that did not match any deployment.
    pub unmatched_subsets: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Return `true` when every key/value in `selector` also appears in `labels`.
pub fn match_labels_subset(
    selector: &IndexMap<String, String>,
    labels: &IndexMap<String, String>,
) -> bool {
    selector
        .iter()
        .all(|(k, v)| labels.get(k).map_or(false, |lv| lv == v))
}

/// Try to extract `(service_name, namespace)` from a Kubernetes FQDN.
///
/// Accepted forms:
/// - `name.namespace.svc.cluster.local`
/// - `name.namespace.svc`
pub fn parse_fqdn(host: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() >= 4 && parts[2] == "svc" {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else if parts.len() == 3 && parts[2] == "svc" {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

/// Filter `KubeService` references out of a heterogeneous resource list.
pub fn collect_services(resources: &[KubernetesResource]) -> Vec<&KubeService> {
    resources
        .iter()
        .filter_map(|r| match r {
            KubernetesResource::Service(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// Filter `Deployment` references out of a heterogeneous resource list.
pub fn collect_deployments(resources: &[KubernetesResource]) -> Vec<&Deployment> {
    resources
        .iter()
        .filter_map(|r| match r {
            KubernetesResource::Deployment(d) => Some(d),
            _ => None,
        })
        .collect()
}

/// Derive `ServiceEndpoint` entries from the ports defined on a Kubernetes
/// `Service`.  The address is synthesised as `<name>.<namespace>.svc.cluster.local`.
pub fn build_service_endpoint(svc: &KubeService) -> Vec<ServiceEndpoint> {
    let address = format!(
        "{}.{}.svc.cluster.local",
        svc.metadata.name, svc.metadata.namespace
    );
    svc.spec
        .ports
        .iter()
        .map(|p| ServiceEndpoint {
            address: address.clone(),
            port: p.port,
            protocol: p.protocol.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ReferenceResolver
// ---------------------------------------------------------------------------

/// Holds all collected Kubernetes and Istio resources and resolves the
/// cross-resource references between them.
#[derive(Debug, Default)]
pub struct ReferenceResolver {
    kubernetes_resources: Vec<KubernetesResource>,
    istio_configs: Vec<IstioConfig>,
}

impl ReferenceResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_kubernetes_resource(&mut self, resource: KubernetesResource) {
        self.kubernetes_resources.push(resource);
    }

    pub fn add_istio_config(&mut self, config: IstioConfig) {
        self.istio_configs.push(config);
    }

    // -- Full resolution ------------------------------------------------

    /// Run a complete resolution pass: service→deployment mapping, Istio
    /// host resolution, dependency inference, and policy extraction.
    pub fn resolve_all(&self) -> Result<ResolvedRefs> {
        let mut refs = self
            .resolve_service_references()
            .context("resolving k8s service references")?;

        // Resolve Istio configs against discovered services.
        let services: Vec<&KubeService> = collect_services(&self.kubernetes_resources);
        let deployments: Vec<&Deployment> = collect_deployments(&self.kubernetes_resources);

        for cfg in &self.istio_configs {
            match cfg {
                IstioConfig::VirtualService(vs) => {
                    let svc_slice: Vec<KubeService> =
                        services.iter().map(|s| (*s).clone()).collect();
                    match Self::resolve_virtual_service_hosts(vs, &svc_slice) {
                        Ok(ids) => {
                            for id in &ids {
                                if let Some(rs) = refs.service_map.get_mut(id) {
                                    rs.virtual_services.push(vs.metadata.name.clone());
                                }
                            }
                            // Extract policy from first HTTP route.
                            for id in &ids {
                                self.extract_vs_policies(vs, id, &mut refs);
                            }
                            // Dependency edges from route destinations.
                            self.add_vs_dependency_edges(vs, &svc_slice, &mut refs);
                        }
                        Err(e) => {
                            log::warn!("VS {} host resolution failed: {}", vs.metadata.name, e);
                        }
                    }
                }
                IstioConfig::DestinationRule(dr) => {
                    let svc_slice: Vec<KubeService> =
                        services.iter().map(|s| (*s).clone()).collect();
                    match Self::resolve_destination_rule_host(dr, &svc_slice) {
                        Ok(id) => {
                            if let Some(rs) = refs.service_map.get_mut(&id) {
                                rs.destination_rules.push(dr.metadata.name.clone());
                            }
                            self.extract_dr_policies(dr, &id, &mut refs);
                        }
                        Err(_) => {
                            refs.unresolved.push(UnresolvedRef {
                                source_kind: "DestinationRule".into(),
                                source_name: dr.metadata.name.clone(),
                                target_host: dr.host.clone(),
                                reason: "no matching Kubernetes service".into(),
                            });
                        }
                    }

                    // Subset routing.
                    let deploy_slice: Vec<Deployment> =
                        deployments.iter().map(|d| (*d).clone()).collect();
                    if let Ok(sr) = Self::resolve_subset_routing(dr, &deploy_slice) {
                        for subset_name in &sr.unmatched_subsets {
                            log::warn!(
                                "DR {} subset '{}' has no matching deployment",
                                dr.metadata.name,
                                subset_name
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // Dependency graph.
        let dep_edges = self.infer_service_dependencies();
        for edge in dep_edges {
            if !refs.dependency_graph.contains(&edge) {
                refs.dependency_graph.push(edge);
            }
        }

        Ok(refs)
    }

    // -- Service → Deployment resolution --------------------------------

    /// Build the base `ResolvedRefs` by mapping each Kubernetes `Service`
    /// to the `Deployment`(s) whose pod-template labels satisfy the
    /// service's selector.
    pub fn resolve_service_references(&self) -> Result<ResolvedRefs> {
        let services = collect_services(&self.kubernetes_resources);
        let deployments = collect_deployments(&self.kubernetes_resources);

        let mut refs = ResolvedRefs::default();

        for svc in &services {
            let id = ServiceId::new(&svc.metadata.name, &svc.metadata.namespace);

            let matching_deploys: Vec<String> = deployments
                .iter()
                .filter(|d| {
                    d.metadata.namespace == svc.metadata.namespace
                        && match_labels_subset(&svc.spec.selector, &d.spec.template.metadata.labels)
                })
                .map(|d| d.metadata.name.clone())
                .collect();

            let endpoints = build_service_endpoint(svc);
            refs.endpoint_map.insert(id.clone(), endpoints);

            refs.service_map.insert(
                id.clone(),
                ResolvedService {
                    service: id.clone(),
                    kubernetes_service: Some(svc.metadata.name.clone()),
                    deployments: matching_deploys,
                    virtual_services: Vec::new(),
                    destination_rules: Vec::new(),
                },
            );
        }

        Ok(refs)
    }

    // -- VirtualService host resolution ---------------------------------

    /// Resolve the `hosts` list of a `VirtualService` against a set of
    /// Kubernetes services.  Short names, FQDNs, and `*` wildcards are
    /// all handled.
    pub fn resolve_virtual_service_hosts(
        vs: &VirtualService,
        services: &[KubeService],
    ) -> Result<Vec<ServiceId>> {
        let resolver = CrossNamespaceResolver::new(vs.metadata.namespace.clone());
        let mut result: Vec<ServiceId> = Vec::new();

        for host in &vs.hosts {
            if host == "*" {
                // Wildcard matches every service in the VS namespace.
                for svc in services {
                    if svc.metadata.namespace == vs.metadata.namespace {
                        let id = ServiceId::new(&svc.metadata.name, &svc.metadata.namespace);
                        if !result.contains(&id) {
                            result.push(id);
                        }
                    }
                }
                continue;
            }

            if let Some(id) =
                CrossNamespaceResolver::resolve_host(host, &vs.metadata.namespace, services)
            {
                if !result.contains(&id) {
                    result.push(id);
                }
            } else if host.starts_with("*.") {
                // Wildcard namespace pattern like *.ns.svc.cluster.local
                if let Some((_name, ns)) = parse_fqdn(&host.replacen("*.", "wildcard.", 1)) {
                    for svc in services {
                        if svc.metadata.namespace == ns {
                            let id = ServiceId::new(&svc.metadata.name, &ns);
                            if !result.contains(&id) {
                                result.push(id);
                            }
                        }
                    }
                }
            }
        }

        if result.is_empty() {
            bail!(
                "VirtualService '{}' hosts {:?} did not match any known service",
                vs.metadata.name,
                vs.hosts
            );
        }

        Ok(result)
    }

    // -- DestinationRule host resolution --------------------------------

    /// Match a `DestinationRule.host` to exactly one Kubernetes service.
    pub fn resolve_destination_rule_host(
        dr: &DestinationRule,
        services: &[KubeService],
    ) -> Result<ServiceId> {
        let ns = &dr.metadata.namespace;
        if let Some(id) = CrossNamespaceResolver::resolve_host(&dr.host, ns, services) {
            Ok(id)
        } else {
            bail!(
                "DestinationRule '{}' host '{}' does not match any service",
                dr.metadata.name,
                dr.host
            )
        }
    }

    // -- Subset routing -------------------------------------------------

    /// For each subset in the `DestinationRule`, find deployments whose
    /// pod-template labels satisfy the subset's label selector and build
    /// synthetic endpoints.
    pub fn resolve_subset_routing(
        dr: &DestinationRule,
        deployments: &[Deployment],
    ) -> Result<SubsetRouting> {
        let mut routing = SubsetRouting::default();

        for subset in &dr.subsets {
            let mut endpoints: Vec<ServiceEndpoint> = Vec::new();

            for deploy in deployments {
                if match_labels_subset(&subset.labels, &deploy.spec.template.metadata.labels) {
                    let addr = format!(
                        "{}.{}.svc.cluster.local",
                        deploy.metadata.name, deploy.metadata.namespace
                    );
                    endpoints.push(ServiceEndpoint {
                        address: addr,
                        port: 80, // placeholder port
                        protocol: "TCP".into(),
                    });
                }
            }

            if endpoints.is_empty() {
                routing.unmatched_subsets.push(subset.name.clone());
            } else {
                routing.subsets.insert(subset.name.clone(), endpoints);
            }
        }

        Ok(routing)
    }

    // -- Dependency inference -------------------------------------------

    /// Build dependency edges by inspecting VirtualService route
    /// destinations, environment variable references in container specs,
    /// and ConfigMap references.
    pub fn infer_service_dependencies(&self) -> Vec<(ServiceId, ServiceId)> {
        let services = collect_services(&self.kubernetes_resources);
        let svc_slice: Vec<KubeService> = services.iter().map(|s| (*s).clone()).collect();
        let mut edges: Vec<(ServiceId, ServiceId)> = Vec::new();

        // (a) VirtualService route destinations.
        for cfg in &self.istio_configs {
            if let IstioConfig::VirtualService(vs) = cfg {
                let source_id = ServiceId::new(&vs.metadata.name, &vs.metadata.namespace);
                for route in &vs.http_routes {
                    for dest in &route.route {
                        if let Some(target) = CrossNamespaceResolver::resolve_host(
                            &dest.destination.host,
                            &vs.metadata.namespace,
                            &svc_slice,
                        ) {
                            let edge = (source_id.clone(), target);
                            if !edges.contains(&edge) {
                                edges.push(edge);
                            }
                        }
                    }
                }
            }
        }

        // (b) Environment variable references in container specs.
        for res in &self.kubernetes_resources {
            if let KubernetesResource::Deployment(deploy) = res {
                let source_id =
                    ServiceId::new(&deploy.metadata.name, &deploy.metadata.namespace);
                let env_hosts = Self::extract_env_service_refs(deploy);
                for host in env_hosts {
                    if let Some(target) = CrossNamespaceResolver::resolve_host(
                        &host,
                        &deploy.metadata.namespace,
                        &svc_slice,
                    ) {
                        let edge = (source_id.clone(), target);
                        if !edges.contains(&edge) {
                            edges.push(edge);
                        }
                    }
                }
            }
        }

        // (c) ConfigMap references – look for service host patterns.
        for res in &self.kubernetes_resources {
            if let KubernetesResource::ConfigMap(cm) = res {
                let refs = Self::extract_configmap_service_refs(cm);
                for (from_deploy, target_host) in refs {
                    if let Some(target) = CrossNamespaceResolver::resolve_host(
                        &target_host,
                        &cm.metadata.namespace,
                        &svc_slice,
                    ) {
                        let edge = (from_deploy, target);
                        if !edges.contains(&edge) {
                            edges.push(edge);
                        }
                    }
                }
            }
        }

        edges
    }

    // -- Private helpers ------------------------------------------------

    fn extract_vs_policies(
        &self,
        vs: &VirtualService,
        id: &ServiceId,
        refs: &mut ResolvedRefs,
    ) {
        if refs.policy_map.contains_key(id) {
            return;
        }

        for route in &vs.http_routes {
            let retry = route.retries.as_ref().map(|r| RetryPolicy {
                max_retries: r.attempts,
                per_try_timeout_ms: parse_duration_ms(&r.per_try_timeout),
                retry_on: r.retry_on.split(',').map(|s| s.trim().to_string()).collect(),
                backoff_base_ms: 25,
                backoff_max_ms: 250,
            });

            let timeout = route.timeout.as_ref().map(|t| TimeoutPolicy {
                request_timeout_ms: parse_duration_ms(t),
                idle_timeout_ms: TimeoutPolicy::default().idle_timeout_ms,
                connect_timeout_ms: TimeoutPolicy::default().connect_timeout_ms,
            });

            if retry.is_some() || timeout.is_some() {
                refs.policy_map.insert(
                    id.clone(),
                    ResolvedPolicies {
                        retry_policy: retry,
                        timeout_policy: timeout,
                        source: PolicySource::Istio,
                    },
                );
                return;
            }
        }
    }

    fn extract_dr_policies(
        &self,
        dr: &DestinationRule,
        id: &ServiceId,
        refs: &mut ResolvedRefs,
    ) {
        if let Some(tp) = &dr.traffic_policy {
            let timeout = tp.connection_pool.as_ref().map(|cp| {
                let connect_ms = cp.tcp.as_ref()
                    .map(|t| parse_duration_ms(&t.connect_timeout))
                    .unwrap_or(5000);
                let idle_ms = cp.http.as_ref()
                    .map(|h| parse_duration_ms(&h.idle_timeout))
                    .unwrap_or(300_000);
                TimeoutPolicy {
                    request_timeout_ms: connect_ms,
                    idle_timeout_ms: idle_ms,
                    connect_timeout_ms: connect_ms,
                }
            });

            let retry = tp.outlier_detection.as_ref().map(|od| {
                let interval_ms = parse_duration_ms(&od.interval);
                let base_ejection_ms = parse_duration_ms(&od.base_ejection_time);
                RetryPolicy {
                    max_retries: od.consecutive_errors,
                    per_try_timeout_ms: interval_ms,
                    retry_on: vec!["5xx".into(), "gateway-error".into()],
                    backoff_base_ms: base_ejection_ms,
                    backoff_max_ms: base_ejection_ms * 3,
                }
            });

            if timeout.is_some() || retry.is_some() {
                let existing = refs.policy_map.get(id);
                let source = if existing.is_some() {
                    PolicySource::Merged
                } else {
                    PolicySource::Istio
                };

                refs.policy_map.insert(
                    id.clone(),
                    ResolvedPolicies {
                        retry_policy: retry.or_else(|| {
                            existing.and_then(|e| e.retry_policy.clone())
                        }),
                        timeout_policy: timeout.or_else(|| {
                            existing.and_then(|e| e.timeout_policy.clone())
                        }),
                        source,
                    },
                );
            }
        }
    }

    fn add_vs_dependency_edges(
        &self,
        vs: &VirtualService,
        services: &[KubeService],
        refs: &mut ResolvedRefs,
    ) {
        // Try to find the source service that the VS sits in front of.
        let source_candidates: Vec<ServiceId> = vs
            .hosts
            .iter()
            .filter_map(|h| {
                CrossNamespaceResolver::resolve_host(h, &vs.metadata.namespace, services)
            })
            .collect();

        let source = source_candidates
            .first()
            .cloned()
            .unwrap_or_else(|| ServiceId::new(&vs.metadata.name, &vs.metadata.namespace));

        for route in &vs.http_routes {
            for dest in &route.route {
                if let Some(target) = CrossNamespaceResolver::resolve_host(
                    &dest.destination.host,
                    &vs.metadata.namespace,
                    services,
                ) {
                    if target != source {
                        let edge = (source.clone(), target);
                        if !refs.dependency_graph.contains(&edge) {
                            refs.dependency_graph.push(edge);
                        }
                    }
                } else {
                    refs.unresolved.push(UnresolvedRef {
                        source_kind: "VirtualService".into(),
                        source_name: vs.metadata.name.clone(),
                        target_host: dest.destination.host.clone(),
                        reason: "destination host not found".into(),
                    });
                }
            }
        }
    }

    /// Scan deployment container env vars for values that look like
    /// Kubernetes service hostnames.
    fn extract_env_service_refs(deploy: &Deployment) -> Vec<String> {
        let mut hosts = Vec::new();
        for container in &deploy.spec.template.containers {
            for env in &container.env {
                if let Some(val) = &env.value {
                    // Match patterns like http://svc-name.namespace:port or
                    // svc-name.namespace.svc.cluster.local
                    for candidate in extract_hostnames_from_value(val) {
                        if !hosts.contains(&candidate) {
                            hosts.push(candidate);
                        }
                    }
                }
            }
        }
        hosts
    }

    /// Scan ConfigMap data values for service hostname patterns.
    fn extract_configmap_service_refs(
        cm: &crate::kubernetes::ConfigMap,
    ) -> Vec<(ServiceId, String)> {
        let mut refs = Vec::new();
        let source = ServiceId::new(&cm.metadata.name, &cm.metadata.namespace);
        for (_key, value) in &cm.data {
            for host in extract_hostnames_from_value(value) {
                refs.push((source.clone(), host));
            }
        }
        refs
    }
}

/// Parse a duration string like "5s", "200ms", "1.5s" into milliseconds.
fn parse_duration_ms(s: &str) -> u64 {
    let s = s.trim();
    if let Some(ms_str) = s.strip_suffix("ms") {
        ms_str.parse::<f64>().unwrap_or(0.0) as u64
    } else if let Some(s_str) = s.strip_suffix('s') {
        (s_str.parse::<f64>().unwrap_or(0.0) * 1000.0) as u64
    } else if let Some(m_str) = s.strip_suffix('m') {
        (m_str.parse::<f64>().unwrap_or(0.0) * 60_000.0) as u64
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

/// Best-effort hostname extraction from a string value.
fn extract_hostnames_from_value(value: &str) -> Vec<String> {
    let mut hosts = Vec::new();
    // Match svc.cluster.local FQDNs.
    let fqdn_re = regex::Regex::new(
        r"([a-z0-9][-a-z0-9]*\.[a-z0-9][-a-z0-9]*\.svc(?:\.cluster\.local)?)",
    )
    .expect("valid regex");
    for cap in fqdn_re.captures_iter(value) {
        hosts.push(cap[1].to_string());
    }

    // Match URL-style references: scheme://host(:port)?
    let url_re =
        regex::Regex::new(r"https?://([a-z0-9][-a-z0-9.]*)(?::\d+)?").expect("valid regex");
    for cap in url_re.captures_iter(value) {
        let h = cap[1].to_string();
        if !hosts.contains(&h) && h.contains('.') {
            hosts.push(h);
        }
    }

    hosts
}

// ---------------------------------------------------------------------------
// CrossNamespaceResolver
// ---------------------------------------------------------------------------

/// Resolves service host strings across namespaces.
pub struct CrossNamespaceResolver {
    default_namespace: String,
}

impl CrossNamespaceResolver {
    pub fn new(default_namespace: String) -> Self {
        Self { default_namespace }
    }

    /// Resolve a host string to a `ServiceId`.
    ///
    /// Resolution order:
    /// 1. FQDN (`name.namespace.svc.cluster.local`)
    /// 2. Two-part (`name.namespace`)
    /// 3. Short name – search `source_namespace` first, then other namespaces
    /// 4. Wildcard prefix (`*.namespace.svc.cluster.local`)
    pub fn resolve_host(
        host: &str,
        source_namespace: &str,
        services: &[KubeService],
    ) -> Option<ServiceId> {
        // 1. FQDN
        if let Some((name, ns)) = parse_fqdn(host) {
            return services
                .iter()
                .find(|s| s.metadata.name == name && s.metadata.namespace == ns)
                .map(|s| ServiceId::new(&s.metadata.name, &s.metadata.namespace));
        }

        let parts: Vec<&str> = host.split('.').collect();

        // 2. Two-part: name.namespace
        if parts.len() == 2 {
            let (name, ns) = (parts[0], parts[1]);
            return services
                .iter()
                .find(|s| s.metadata.name == name && s.metadata.namespace == ns)
                .map(|s| ServiceId::new(&s.metadata.name, &s.metadata.namespace));
        }

        // 3. Short name
        if parts.len() == 1 {
            let name = parts[0];
            // Prefer source namespace.
            if let Some(svc) = services
                .iter()
                .find(|s| s.metadata.name == name && s.metadata.namespace == source_namespace)
            {
                return Some(ServiceId::new(&svc.metadata.name, &svc.metadata.namespace));
            }
            // Fall back to any namespace.
            if let Some(svc) = services.iter().find(|s| s.metadata.name == name) {
                return Some(ServiceId::new(&svc.metadata.name, &svc.metadata.namespace));
            }
        }

        // 4. Wildcard
        if host.starts_with("*.") {
            let replaced = host.replacen("*.", "wildcard.", 1);
            if let Some((_name, ns)) = parse_fqdn(&replaced) {
                return services
                    .iter()
                    .find(|s| s.metadata.namespace == ns)
                    .map(|s| ServiceId::new(&s.metadata.name, &s.metadata.namespace));
            }
        }

        None
    }

    /// Returns `true` if `host` refers to a mesh-internal address (i.e., a
    /// short name or a `.svc.cluster.local` FQDN).
    pub fn is_mesh_internal(host: &str) -> bool {
        if host.contains("svc.cluster.local") || host.ends_with(".svc") {
            return true;
        }
        // Short names (no dots at all, or only name.namespace with no further
        // TLD segments) are mesh-internal.
        let dot_count = host.chars().filter(|c| *c == '.').count();
        dot_count <= 1
    }

    /// Normalise any host format to a FQDN
    /// (`name.namespace.svc.cluster.local`).
    pub fn normalize_host(host: &str, source_namespace: &str) -> String {
        // Already FQDN?
        if host.ends_with(".svc.cluster.local") {
            return host.to_string();
        }

        // Ends with .svc  (e.g. name.ns.svc)
        if host.ends_with(".svc") {
            return format!("{}.cluster.local", host);
        }

        let parts: Vec<&str> = host.split('.').collect();

        match parts.len() {
            1 => {
                // Short name → use source namespace.
                format!("{}.{}.svc.cluster.local", host, source_namespace)
            }
            2 => {
                // name.namespace
                format!("{}.svc.cluster.local", host)
            }
            _ => {
                // Unknown form – return as-is.
                host.to_string()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DefaultValueInferrer
// ---------------------------------------------------------------------------

/// Infers sensible default retry and timeout policies for services that
/// lack explicit configuration, using the service's position in the
/// dependency graph.
pub struct DefaultValueInferrer;

impl DefaultValueInferrer {
    pub fn new() -> Self {
        Self
    }

    /// Infer a retry policy for `service_id`.
    ///
    /// Leaf services (no downstream deps) get more retries because the
    /// blast radius is contained.  Gateway / edge services that fan out
    /// to many backends get fewer retries to avoid retry storms.
    pub fn infer_retry_policy(
        service_id: &ServiceId,
        resolved: &ResolvedRefs,
    ) -> RetryPolicy {
        let fan_out = Self::compute_fan_out(service_id, &resolved.dependency_graph);
        let depth = Self::compute_depth(service_id, &resolved.dependency_graph);
        let budget = Self::suggest_retry_budget(fan_out, depth);

        let per_try_timeout = if depth == 0 {
            2000 // leaf services – generous per-try timeout
        } else {
            1000u64.saturating_sub(depth as u64 * 100).max(200)
        };

        RetryPolicy {
            max_retries: budget,
            per_try_timeout_ms: per_try_timeout,
            retry_on: vec!["5xx".into(), "reset".into(), "connect-failure".into()],
            backoff_base_ms: 25,
            backoff_max_ms: 250 * (depth as u64 + 1),
        }
    }

    /// Infer a timeout policy for `service_id`.
    ///
    /// Edge / gateway services get longer request timeouts to account for
    /// fan-out latency.  Leaf services can have tighter timeouts.
    pub fn infer_timeout_policy(
        service_id: &ServiceId,
        resolved: &ResolvedRefs,
    ) -> TimeoutPolicy {
        let fan_out = Self::compute_fan_out(service_id, &resolved.dependency_graph);
        let depth = Self::compute_depth(service_id, &resolved.dependency_graph);

        let base_timeout: u64 = 5000;
        let request_timeout = base_timeout + (fan_out as u64 * 2000) + (depth as u64 * 1500);
        let connect_timeout = if depth == 0 { 3000 } else { 5000 };
        let idle_timeout = 300_000u64; // 5 min default

        TimeoutPolicy {
            request_timeout_ms: request_timeout.min(60_000),
            idle_timeout_ms: idle_timeout,
            connect_timeout_ms: connect_timeout,
        }
    }

    /// Count the number of direct downstream dependencies of a service.
    pub fn compute_fan_out(
        service_id: &ServiceId,
        deps: &[(ServiceId, ServiceId)],
    ) -> usize {
        deps.iter()
            .filter(|(src, _dst)| src == service_id)
            .map(|(_src, dst)| dst.clone())
            .collect::<HashSet<_>>()
            .len()
    }

    /// BFS depth from `service_id` to any leaf node (a node with no
    /// outgoing edges).  Returns 0 if the service itself is a leaf.
    pub fn compute_depth(
        service_id: &ServiceId,
        deps: &[(ServiceId, ServiceId)],
    ) -> usize {
        let mut visited: HashSet<ServiceId> = HashSet::new();
        let mut queue: VecDeque<(ServiceId, usize)> = VecDeque::new();
        queue.push_back((service_id.clone(), 0));
        visited.insert(service_id.clone());

        let mut max_depth: usize = 0;

        while let Some((current, depth)) = queue.pop_front() {
            let children: Vec<&ServiceId> = deps
                .iter()
                .filter(|(src, _)| *src == current)
                .map(|(_, dst)| dst)
                .collect();

            if children.is_empty() {
                max_depth = max_depth.max(depth);
            } else {
                for child in children {
                    if !visited.contains(child) {
                        visited.insert(child.clone());
                        queue.push_back((child.clone(), depth + 1));
                    }
                }
            }
        }

        max_depth
    }

    /// Suggest a retry budget.  High fan-out or deep position means
    /// fewer retries to prevent amplification.
    pub fn suggest_retry_budget(fan_out: usize, depth: usize) -> u32 {
        let base: u32 = 5;
        let penalty = (fan_out as u32).saturating_mul(1) + (depth as u32).saturating_mul(1);
        base.saturating_sub(penalty).max(1)
    }
}

impl Default for DefaultValueInferrer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::istio::{
        DestinationRule, Destination, HttpRoute, HttpRouteDestination, IstioConfig, Subset,
        VirtualService,
    };
    use crate::kubernetes::{
        ContainerSpec, Deployment, DeploymentSpec, KubeService, KubernetesResource, ServicePort,
        ServiceSpec,
    };
    use crate::{ObjectMeta, ServiceId};
    use indexmap::IndexMap;

    // -- Factories ------------------------------------------------------

    fn make_meta(name: &str, namespace: &str) -> ObjectMeta {
        ObjectMeta {
            name: name.into(),
            namespace: namespace.into(),
            labels: IndexMap::new(),
            annotations: IndexMap::new(),
            uid: String::new(),
            resource_version: String::new(),
        }
    }

    fn make_service(name: &str, namespace: &str, selector: IndexMap<String, String>) -> KubeService {
        KubeService {
            metadata: make_meta(name, namespace),
            spec: ServiceSpec {
                service_type: "ClusterIP".into(),
                selector,
                ports: vec![ServicePort {
                    name: Some("http".into()),
                    port: 80,
                    target_port: 8080,
                    protocol: "TCP".into(),
                    node_port: None,
                }],
                cluster_ip: None,
            },
        }
    }

    fn make_deployment(
        name: &str,
        namespace: &str,
        pod_labels: IndexMap<String, String>,
    ) -> Deployment {
        Deployment {
            metadata: make_meta(name, namespace),
            spec: DeploymentSpec {
                replicas: 1,
                selector: crate::kubernetes::LabelSelector {
                    match_labels: pod_labels.clone(),
                    match_expressions: Vec::new(),
                },
                template: crate::kubernetes::PodTemplateSpec {
                    metadata: crate::ObjectMeta {
                        labels: pod_labels,
                        ..Default::default()
                    },
                    containers: Vec::new(),
                    init_containers: Vec::new(),
                    volumes: Vec::new(),
                    service_account: None,
                },
                strategy: crate::kubernetes::DeploymentStrategy::default(),
            },
        }
    }

    fn make_virtual_service(
        name: &str,
        namespace: &str,
        hosts: Vec<&str>,
        route_hosts: Vec<&str>,
    ) -> VirtualService {
        VirtualService {
            metadata: make_meta(name, namespace),
            hosts: hosts.into_iter().map(String::from).collect(),
            gateways: Vec::new(),
            http_routes: vec![HttpRoute {
                name: Some("default".into()),
                match_conditions: Vec::new(),
                route: route_hosts
                    .into_iter()
                    .map(|h| HttpRouteDestination {
                        destination: Destination {
                            host: h.into(),
                            port: None,
                            subset: None,
                        },
                        weight: 100,
                        headers: None,
                    })
                    .collect(),
                retries: None,
                timeout: None,
                fault: None,
                mirror: None,
                headers: None,
                rewrite: None,
            }],
            tcp_routes: Vec::new(),
            tls_routes: Vec::new(),
            export_to: Vec::new(),
        }
    }

    fn make_destination_rule(
        name: &str,
        namespace: &str,
        host: &str,
        subsets: Vec<(&str, IndexMap<String, String>)>,
    ) -> DestinationRule {
        DestinationRule {
            metadata: make_meta(name, namespace),
            host: host.into(),
            traffic_policy: None,
            subsets: subsets
                .into_iter()
                .map(|(sn, labels)| Subset {
                    name: sn.into(),
                    labels,
                    traffic_policy: None,
                })
                .collect(),
            export_to: Vec::new(),
        }
    }

    // -- Tests ----------------------------------------------------------

    #[test]
    fn test_resolve_service_to_deployment() {
        let mut labels = IndexMap::new();
        labels.insert("app".into(), "frontend".into());

        let svc = make_service("frontend-svc", "default", labels.clone());
        let deploy = make_deployment("frontend-deploy", "default", labels);

        let mut resolver = ReferenceResolver::new();
        resolver.add_kubernetes_resource(KubernetesResource::Service(svc));
        resolver.add_kubernetes_resource(KubernetesResource::Deployment(deploy));

        let refs = resolver.resolve_service_references().unwrap();
        let id = ServiceId::new("frontend-svc", "default");
        let rs = refs.service_map.get(&id).expect("service should exist");
        assert_eq!(rs.deployments, vec!["frontend-deploy"]);
    }

    #[test]
    fn test_resolve_virtual_service_host_short_name() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "api".into());
        let svc = make_service("api", "default", sel);

        let vs = make_virtual_service("api-vs", "default", vec!["api"], vec!["api"]);

        let ids = ReferenceResolver::resolve_virtual_service_hosts(&vs, &[svc]).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].name, "api");
        assert_eq!(ids[0].namespace, "default");
    }

    #[test]
    fn test_resolve_virtual_service_host_fqdn() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "payments".into());
        let svc = make_service("payments", "finance", sel);

        let vs = make_virtual_service(
            "pay-vs",
            "finance",
            vec!["payments.finance.svc.cluster.local"],
            vec!["payments"],
        );

        let ids = ReferenceResolver::resolve_virtual_service_hosts(&vs, &[svc]).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].name, "payments");
        assert_eq!(ids[0].namespace, "finance");
    }

    #[test]
    fn test_resolve_destination_rule_host() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "orders".into());
        let svc = make_service("orders", "default", sel);

        let dr = make_destination_rule("orders-dr", "default", "orders", vec![]);

        let id = ReferenceResolver::resolve_destination_rule_host(&dr, &[svc]).unwrap();
        assert_eq!(id.name, "orders");
    }

    #[test]
    fn test_resolve_subset_routing() {
        let mut v1_labels = IndexMap::new();
        v1_labels.insert("version".into(), "v1".into());
        let mut v2_labels = IndexMap::new();
        v2_labels.insert("version".into(), "v2".into());
        let mut v3_labels = IndexMap::new();
        v3_labels.insert("version".into(), "v3".into());

        let d1 = make_deployment("api-v1", "default", v1_labels.clone());
        let d2 = make_deployment("api-v2", "default", v2_labels.clone());

        let dr = make_destination_rule(
            "api-dr",
            "default",
            "api",
            vec![("v1", v1_labels), ("v2", v2_labels), ("v3", v3_labels)],
        );

        let routing = ReferenceResolver::resolve_subset_routing(&dr, &[d1, d2]).unwrap();
        assert!(routing.subsets.contains_key("v1"));
        assert!(routing.subsets.contains_key("v2"));
        assert_eq!(routing.unmatched_subsets, vec!["v3"]);
    }

    #[test]
    fn test_infer_dependencies_from_vs() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "backend".into());
        let svc = make_service("backend", "default", sel);

        let vs = make_virtual_service(
            "gateway-vs",
            "default",
            vec!["gateway-vs"],
            vec!["backend"],
        );

        let mut resolver = ReferenceResolver::new();
        resolver.add_kubernetes_resource(KubernetesResource::Service(svc));
        resolver.add_istio_config(IstioConfig::VirtualService(vs));

        let edges = resolver.infer_service_dependencies();
        assert!(!edges.is_empty());
        let (src, dst) = &edges[0];
        assert_eq!(src.name, "gateway-vs");
        assert_eq!(dst.name, "backend");
    }

    #[test]
    fn test_cross_namespace_short_name() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "cache".into());
        let svc = make_service("cache", "infra", sel);

        let id =
            CrossNamespaceResolver::resolve_host("cache", "infra", &[svc]).unwrap();
        assert_eq!(id.name, "cache");
        assert_eq!(id.namespace, "infra");
    }

    #[test]
    fn test_cross_namespace_fqdn() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "db".into());
        let svc = make_service("db", "storage", sel);

        let id = CrossNamespaceResolver::resolve_host(
            "db.storage.svc.cluster.local",
            "default",
            &[svc],
        )
        .unwrap();
        assert_eq!(id.name, "db");
        assert_eq!(id.namespace, "storage");
    }

    #[test]
    fn test_normalize_host() {
        assert_eq!(
            CrossNamespaceResolver::normalize_host("api", "default"),
            "api.default.svc.cluster.local"
        );
        assert_eq!(
            CrossNamespaceResolver::normalize_host("api.prod", "default"),
            "api.prod.svc.cluster.local"
        );
        assert_eq!(
            CrossNamespaceResolver::normalize_host("api.prod.svc.cluster.local", "default"),
            "api.prod.svc.cluster.local"
        );
        assert_eq!(
            CrossNamespaceResolver::normalize_host("api.prod.svc", "default"),
            "api.prod.svc.cluster.local"
        );
    }

    #[test]
    fn test_is_mesh_internal() {
        assert!(CrossNamespaceResolver::is_mesh_internal("api"));
        assert!(CrossNamespaceResolver::is_mesh_internal("api.default"));
        assert!(CrossNamespaceResolver::is_mesh_internal(
            "api.default.svc.cluster.local"
        ));
        assert!(CrossNamespaceResolver::is_mesh_internal("api.default.svc"));
        assert!(!CrossNamespaceResolver::is_mesh_internal(
            "external.example.com"
        ));
    }

    #[test]
    fn test_default_retry_inference() {
        let mut refs = ResolvedRefs::default();
        // leaf service
        let leaf = ServiceId::new("leaf-svc", "default");
        refs.service_map.insert(
            leaf.clone(),
            ResolvedService {
                service: leaf.clone(),
                kubernetes_service: Some("leaf-svc".into()),
                deployments: vec!["leaf-deploy".into()],
                virtual_services: Vec::new(),
                destination_rules: Vec::new(),
            },
        );

        let policy = DefaultValueInferrer::infer_retry_policy(&leaf, &refs);
        // Leaf (fan_out=0, depth=0) → budget = max(5 - 0, 1) = 5
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.per_try_timeout_ms, 2000);
    }

    #[test]
    fn test_fan_out_calculation() {
        let a = ServiceId::new("a", "default");
        let b = ServiceId::new("b", "default");
        let c = ServiceId::new("c", "default");

        let deps = vec![
            (a.clone(), b.clone()),
            (a.clone(), c.clone()),
            (b.clone(), c.clone()),
        ];

        assert_eq!(DefaultValueInferrer::compute_fan_out(&a, &deps), 2);
        assert_eq!(DefaultValueInferrer::compute_fan_out(&b, &deps), 1);
        assert_eq!(DefaultValueInferrer::compute_fan_out(&c, &deps), 0);
    }

    #[test]
    fn test_depth_calculation() {
        let a = ServiceId::new("a", "default");
        let b = ServiceId::new("b", "default");
        let c = ServiceId::new("c", "default");

        let deps = vec![
            (a.clone(), b.clone()),
            (b.clone(), c.clone()),
        ];

        assert_eq!(DefaultValueInferrer::compute_depth(&a, &deps), 2);
        assert_eq!(DefaultValueInferrer::compute_depth(&b, &deps), 1);
        assert_eq!(DefaultValueInferrer::compute_depth(&c, &deps), 0);
    }

    #[test]
    fn test_match_labels_subset() {
        let mut selector = IndexMap::new();
        selector.insert("app".into(), "web".into());

        let mut labels = IndexMap::new();
        labels.insert("app".into(), "web".into());
        labels.insert("version".into(), "v1".into());

        assert!(match_labels_subset(&selector, &labels));

        let mut bad_labels = IndexMap::new();
        bad_labels.insert("app".into(), "api".into());
        assert!(!match_labels_subset(&selector, &bad_labels));

        let empty_labels: IndexMap<String, String> = IndexMap::new();
        assert!(!match_labels_subset(&selector, &empty_labels));

        let empty_selector: IndexMap<String, String> = IndexMap::new();
        assert!(match_labels_subset(&empty_selector, &labels));
    }

    #[test]
    fn test_parse_fqdn() {
        let (name, ns) = parse_fqdn("api.prod.svc.cluster.local").unwrap();
        assert_eq!(name, "api");
        assert_eq!(ns, "prod");

        let (name2, ns2) = parse_fqdn("db.storage.svc").unwrap();
        assert_eq!(name2, "db");
        assert_eq!(ns2, "storage");

        assert!(parse_fqdn("shortname").is_none());
        assert!(parse_fqdn("two.parts").is_none());
    }

    #[test]
    fn test_build_service_endpoint() {
        let mut sel = IndexMap::new();
        sel.insert("app".into(), "web".into());
        let svc = make_service("web", "default", sel);

        let eps = build_service_endpoint(&svc);
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].address, "web.default.svc.cluster.local");
        assert_eq!(eps[0].port, 80);
        assert_eq!(eps[0].protocol, "TCP");
    }

    #[test]
    fn test_parse_duration_ms() {
        assert_eq!(parse_duration_ms("5s"), 5000);
        assert_eq!(parse_duration_ms("200ms"), 200);
        assert_eq!(parse_duration_ms("1.5s"), 1500);
        assert_eq!(parse_duration_ms("2m"), 120_000);
    }

    #[test]
    fn test_extract_hostnames_from_value() {
        let hosts = extract_hostnames_from_value(
            "http://api.default.svc.cluster.local:8080/v1",
        );
        assert!(hosts.iter().any(|h| h.contains("api.default.svc")));

        let hosts2 = extract_hostnames_from_value("BACKEND_URL=backend.prod.svc.cluster.local");
        assert!(hosts2.iter().any(|h| h == "backend.prod.svc.cluster.local"));
    }

    #[test]
    fn test_suggest_retry_budget() {
        // No fan-out, no depth → max budget
        assert_eq!(DefaultValueInferrer::suggest_retry_budget(0, 0), 5);
        // High fan-out → reduced
        assert_eq!(DefaultValueInferrer::suggest_retry_budget(3, 0), 2);
        // High depth → reduced
        assert_eq!(DefaultValueInferrer::suggest_retry_budget(0, 4), 1);
        // Both → floor at 1
        assert_eq!(DefaultValueInferrer::suggest_retry_budget(5, 5), 1);
    }
}
