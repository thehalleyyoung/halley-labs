//! Configuration validation for Kubernetes, Istio, and cross-resource consistency.
//!
//! Provides [`ConfigValidator`] which checks resource configurations for common
//! issues such as out-of-range retry counts, inconsistent timeouts, missing
//! policies, circular dependencies, and orphaned resources.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::istio::{
    DestinationRule, Gateway, HttpRoute, IstioConfig, ServiceEntry,
    VirtualService,
};
use crate::kubernetes::{
    ContainerSpec, Deployment, Ingress, KubeService, KubernetesResource,
};
use crate::reference_resolver::ResolvedRefs;
use crate::{RetryPolicy, ServiceId, Severity};

// ---------------------------------------------------------------------------
// ValidationIssue
// ---------------------------------------------------------------------------

/// A single validation finding with severity, location, and optional fix suggestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub resource_kind: String,
    pub resource_name: String,
    pub resource_namespace: String,
    pub field: String,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    pub fn new(
        severity: Severity,
        kind: impl Into<String>,
        name: impl Into<String>,
        namespace: impl Into<String>,
        field: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            resource_kind: kind.into(),
            resource_name: name.into(),
            resource_namespace: namespace.into(),
            field: field.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn info(
        kind: impl Into<String>,
        name: impl Into<String>,
        ns: impl Into<String>,
        field: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Info, kind, name, ns, field, msg)
    }

    pub fn warning(
        kind: impl Into<String>,
        name: impl Into<String>,
        ns: impl Into<String>,
        field: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Warning, kind, name, ns, field, msg)
    }

    pub fn error(
        kind: impl Into<String>,
        name: impl Into<String>,
        ns: impl Into<String>,
        field: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Error, kind, name, ns, field, msg)
    }

    pub fn critical(
        kind: impl Into<String>,
        name: impl Into<String>,
        ns: impl Into<String>,
        field: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Critical, kind, name, ns, field, msg)
    }
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}/{} {}: {} (field: {})",
            self.severity,
            self.resource_namespace,
            self.resource_name,
            self.resource_kind,
            self.message,
            self.field,
        )?;
        if let Some(ref s) = self.suggestion {
            write!(f, " — suggestion: {}", s)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ValidatorConfig
// ---------------------------------------------------------------------------

/// Configurable thresholds and feature-flags for the validator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    pub max_retry_attempts: u32,
    pub min_retry_attempts: u32,
    pub max_timeout_ms: u64,
    pub min_timeout_ms: u64,
    pub max_outlier_ejection_percent: u32,
    pub require_retry_policy: bool,
    pub require_timeout_policy: bool,
    pub require_resource_limits: bool,
    pub max_cascade_depth: usize,
    pub allowed_retry_conditions: Vec<String>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 5,
            min_retry_attempts: 1,
            max_timeout_ms: 60_000,
            min_timeout_ms: 100,
            max_outlier_ejection_percent: 50,
            require_retry_policy: true,
            require_timeout_policy: true,
            require_resource_limits: true,
            max_cascade_depth: 5,
            allowed_retry_conditions: vec![
                "5xx".into(),
                "gateway-error".into(),
                "connect-failure".into(),
                "retriable-4xx".into(),
                "reset".into(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigValidator
// ---------------------------------------------------------------------------

/// Main validation engine.
pub struct ConfigValidator {
    pub config: ValidatorConfig,
}

impl ConfigValidator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ValidatorConfig::default())
    }

    // ------------------------------------------------------------------
    // Aggregate entry-point
    // ------------------------------------------------------------------

    /// Run every validation check and return all issues found.
    pub fn validate_all(
        &self,
        k8s: &[KubernetesResource],
        istio: &[IstioConfig],
        resolved: &ResolvedRefs,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        issues.extend(self.validate_kubernetes_config(k8s));
        issues.extend(self.validate_istio_config(istio));
        issues.extend(self.validate_cross_resource_consistency(k8s, istio, resolved));
        issues
    }

    // ------------------------------------------------------------------
    // Kubernetes validation
    // ------------------------------------------------------------------

    pub fn validate_kubernetes_config(
        &self,
        configs: &[KubernetesResource],
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        let mut services: Vec<&KubeService> = Vec::new();
        let mut deployments: Vec<&Deployment> = Vec::new();
        let mut service_names: HashMap<(String, String), usize> = HashMap::new();

        for res in configs {
            match res {
                KubernetesResource::Deployment(dep) => {
                    issues.extend(self.validate_deployment(dep));
                    deployments.push(dep);
                }
                KubernetesResource::Service(svc) => {
                    let key = (
                        svc.metadata.name.clone(),
                        svc.metadata.namespace.clone(),
                    );
                    *service_names.entry(key).or_insert(0) += 1;
                    services.push(svc);
                }
                KubernetesResource::Ingress(ing) => {
                    issues.extend(self.validate_ingress(ing, &services));
                }
                _ => {}
            }
        }

        // Check for duplicate service names.
        for ((name, ns), count) in &service_names {
            if *count > 1 {
                issues.push(
                    ValidationIssue::error(
                        "Service",
                        name.as_str(),
                        ns.as_str(),
                        "metadata.name",
                        format!("Duplicate Service name '{name}' in namespace '{ns}' ({count} occurrences)"),
                    )
                    .with_suggestion("Rename one of the duplicate services"),
                );
            }
        }

        // Check that every service selector matches at least one deployment.
        for svc in &services {
            if svc.spec.selector.is_empty() {
                continue;
            }
            let matched = deployments.iter().any(|dep| {
                let pod_labels = &dep.spec.template.metadata.labels;
                svc.spec
                    .selector
                    .iter()
                    .all(|(k, v)| pod_labels.get(k).map_or(false, |pv| pv == v))
            });
            if !matched {
                issues.push(
                    ValidationIssue::warning(
                        "Service",
                        &svc.metadata.name,
                        &svc.metadata.namespace,
                        "spec.selector",
                        format!(
                            "No deployment matches selector {:?}",
                            svc.spec.selector
                        ),
                    )
                    .with_suggestion("Ensure at least one Deployment's pod template labels match this Service's selector"),
                );
            }
        }

        issues
    }

    fn validate_deployment(&self, dep: &Deployment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &dep.metadata.namespace;
        let name = &dep.metadata.name;

        for container in &dep.spec.template.containers {
            issues.extend(validate_container(container, name, ns, &self.config));
        }
        for container in &dep.spec.template.init_containers {
            issues.extend(validate_container(container, name, ns, &self.config));
        }

        if dep.spec.replicas == 0 {
            issues.push(ValidationIssue::warning(
                "Deployment",
                name,
                ns,
                "spec.replicas",
                "Deployment has 0 replicas",
            ));
        }

        issues
    }

    fn validate_ingress(
        &self,
        ing: &Ingress,
        services: &[&KubeService],
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &ing.metadata.namespace;
        let name = &ing.metadata.name;

        for rule in &ing.rules {
            for path in &rule.paths {
                let backend_exists = services.iter().any(|svc| {
                    svc.metadata.name == path.backend.service_name
                        && (svc.metadata.namespace == *ns || ns.is_empty())
                });
                if !backend_exists {
                    issues.push(
                        ValidationIssue::error(
                            "Ingress",
                            name,
                            ns,
                            "spec.rules[].http.paths[].backend",
                            format!(
                                "Backend service '{}' not found",
                                path.backend.service_name
                            ),
                        )
                        .with_suggestion("Create the missing Service or fix the backend reference"),
                    );
                }
            }
        }

        issues
    }

    // ------------------------------------------------------------------
    // Istio validation
    // ------------------------------------------------------------------

    pub fn validate_istio_config(&self, configs: &[IstioConfig]) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        let mut vs_hosts: HashMap<String, Vec<String>> = HashMap::new();

        for cfg in configs {
            match cfg {
                IstioConfig::VirtualService(vs) => {
                    issues.extend(self.validate_virtual_service(vs));
                    for host in &vs.hosts {
                        vs_hosts
                            .entry(host.clone())
                            .or_default()
                            .push(vs.metadata.name.clone());
                    }
                }
                IstioConfig::DestinationRule(dr) => {
                    issues.extend(self.validate_destination_rule(dr));
                }
                IstioConfig::Gateway(gw) => {
                    issues.extend(self.validate_gateway(gw));
                }
                IstioConfig::ServiceEntry(se) => {
                    issues.extend(self.validate_service_entry(se));
                }
            }
        }

        // Check for duplicate VirtualService hosts.
        for (host, names) in &vs_hosts {
            if names.len() > 1 {
                issues.push(
                    ValidationIssue::warning(
                        "VirtualService",
                        &names.join(", "),
                        "",
                        "spec.hosts",
                        format!(
                            "Multiple VirtualServices target the same host '{}': [{}]",
                            host,
                            names.join(", ")
                        ),
                    )
                    .with_suggestion(
                        "Merge the VirtualServices or ensure they have non-overlapping match conditions",
                    ),
                );
            }
        }

        issues
    }

    fn validate_virtual_service(&self, vs: &VirtualService) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &vs.metadata.namespace;
        let name = &vs.metadata.name;

        if vs.hosts.is_empty() {
            issues.push(ValidationIssue::error(
                "VirtualService",
                name,
                ns,
                "spec.hosts",
                "VirtualService has no hosts defined",
            ));
        }

        for (i, route) in vs.http_routes.iter().enumerate() {
            issues.extend(validate_http_route(route, name, ns, &self.config, i));
        }

        issues
    }

    fn validate_destination_rule(&self, dr: &DestinationRule) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &dr.metadata.namespace;
        let name = &dr.metadata.name;

        if dr.host.is_empty() {
            issues.push(ValidationIssue::error(
                "DestinationRule",
                name,
                ns,
                "spec.host",
                "DestinationRule has empty host",
            ));
        }

        for subset in &dr.subsets {
            if subset.labels.is_empty() {
                issues.push(
                    ValidationIssue::warning(
                        "DestinationRule",
                        name,
                        ns,
                        &format!("spec.subsets[{}].labels", subset.name),
                        format!("Subset '{}' has no labels", subset.name),
                    )
                    .with_suggestion("Add labels to the subset so it can match pods"),
                );
            }
        }

        if let Some(ref tp) = dr.traffic_policy {
            if let Some(ref od) = tp.outlier_detection {
                if od.max_ejection_percent > self.config.max_outlier_ejection_percent {
                    issues.push(
                        ValidationIssue::warning(
                            "DestinationRule",
                            name,
                            ns,
                            "spec.trafficPolicy.outlierDetection.maxEjectionPercent",
                            format!(
                                "Outlier ejection percent {} exceeds recommended max {}",
                                od.max_ejection_percent,
                                self.config.max_outlier_ejection_percent
                            ),
                        )
                        .with_suggestion(&format!(
                            "Reduce maxEjectionPercent to at most {}",
                            self.config.max_outlier_ejection_percent
                        )),
                    );
                }
                if od.consecutive_errors == 0 {
                    issues.push(ValidationIssue::warning(
                        "DestinationRule",
                        name,
                        ns,
                        "spec.trafficPolicy.outlierDetection.consecutiveErrors",
                        "consecutiveErrors is 0, outlier detection is effectively disabled",
                    ));
                }
            }
        }

        issues
    }

    fn validate_gateway(&self, gw: &Gateway) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &gw.metadata.namespace;
        let name = &gw.metadata.name;

        if gw.servers.is_empty() {
            issues.push(ValidationIssue::warning(
                "Gateway",
                name,
                ns,
                "spec.servers",
                "Gateway has no servers configured",
            ));
        }

        for (i, server) in gw.servers.iter().enumerate() {
            if server.hosts.is_empty() {
                issues.push(ValidationIssue::error(
                    "Gateway",
                    name,
                    ns,
                    &format!("spec.servers[{}].hosts", i),
                    "Gateway server has no hosts",
                ));
            }
            if server.port.protocol.to_uppercase() == "HTTPS" && server.tls.is_none() {
                issues.push(
                    ValidationIssue::error(
                        "Gateway",
                        name,
                        ns,
                        &format!("spec.servers[{}].tls", i),
                        "HTTPS server is missing TLS configuration",
                    )
                    .with_suggestion("Add TLS settings for HTTPS servers"),
                );
            }
        }

        issues
    }

    fn validate_service_entry(&self, se: &ServiceEntry) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let ns = &se.metadata.namespace;
        let name = &se.metadata.name;

        if se.hosts.is_empty() {
            issues.push(ValidationIssue::error(
                "ServiceEntry",
                name,
                ns,
                "spec.hosts",
                "ServiceEntry has no hosts",
            ));
        }

        if se.ports.is_empty() {
            issues.push(ValidationIssue::warning(
                "ServiceEntry",
                name,
                ns,
                "spec.ports",
                "ServiceEntry has no ports defined",
            ));
        }

        issues
    }

    // ------------------------------------------------------------------
    // Cross-resource consistency
    // ------------------------------------------------------------------

    pub fn validate_cross_resource_consistency(
        &self,
        k8s: &[KubernetesResource],
        istio: &[IstioConfig],
        resolved: &ResolvedRefs,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Collect k8s service names for reference checking.
        let svc_names: HashSet<String> = k8s
            .iter()
            .filter_map(|r| {
                if let KubernetesResource::Service(s) = r {
                    Some(s.metadata.name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Check VirtualService hosts reference existing services.
        for cfg in istio {
            if let IstioConfig::VirtualService(vs) = cfg {
                for host in &vs.hosts {
                    let short_name = host.split('.').next().unwrap_or(host);
                    if !host.contains('*') && !svc_names.contains(short_name) {
                        issues.push(
                            ValidationIssue::warning(
                                "VirtualService",
                                &vs.metadata.name,
                                &vs.metadata.namespace,
                                "spec.hosts",
                                format!(
                                    "VirtualService host '{}' does not match any known Service",
                                    host
                                ),
                            )
                            .with_suggestion("Verify the host matches an existing Kubernetes Service"),
                        );
                    }
                }
            }
            if let IstioConfig::DestinationRule(dr) = cfg {
                let short_name = dr.host.split('.').next().unwrap_or(&dr.host);
                if !svc_names.contains(short_name) {
                    issues.push(
                        ValidationIssue::warning(
                            "DestinationRule",
                            &dr.metadata.name,
                            &dr.metadata.namespace,
                            "spec.host",
                            format!(
                                "DestinationRule host '{}' does not match any known Service",
                                dr.host
                            ),
                        )
                        .with_suggestion("Verify the host matches an existing Kubernetes Service"),
                    );
                }
            }
        }

        // Check missing policies.
        issues.extend(self.check_missing_policies(resolved));

        // Check circular dependencies.
        issues.extend(self.check_circular_dependencies(&resolved.dependency_graph));

        // Check orphaned resources.
        issues.extend(self.check_orphaned_resources(k8s, istio));

        issues
    }

    // ------------------------------------------------------------------
    // Retry bounds
    // ------------------------------------------------------------------

    pub fn check_retry_bounds(
        &self,
        policy: &RetryPolicy,
        resource_name: &str,
        namespace: &str,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if policy.max_retries > self.config.max_retry_attempts {
            issues.push(
                ValidationIssue::warning(
                    "RetryPolicy",
                    resource_name,
                    namespace,
                    "maxRetries",
                    format!(
                        "Retry count {} exceeds recommended maximum {}",
                        policy.max_retries, self.config.max_retry_attempts
                    ),
                )
                .with_suggestion(&format!(
                    "Reduce max_retries to at most {}",
                    self.config.max_retry_attempts
                )),
            );
        }

        if policy.max_retries < self.config.min_retry_attempts {
            issues.push(
                ValidationIssue::warning(
                    "RetryPolicy",
                    resource_name,
                    namespace,
                    "maxRetries",
                    format!(
                        "Retry count {} is below recommended minimum {}",
                        policy.max_retries, self.config.min_retry_attempts
                    ),
                )
                .with_suggestion(&format!(
                    "Increase max_retries to at least {}",
                    self.config.min_retry_attempts
                )),
            );
        }

        for cond in &policy.retry_on {
            if !self.config.allowed_retry_conditions.contains(cond) {
                issues.push(
                    ValidationIssue::info(
                        "RetryPolicy",
                        resource_name,
                        namespace,
                        "retryOn",
                        format!("Retry condition '{}' is not in the standard allowed list", cond),
                    )
                    .with_suggestion("Use standard retry conditions: 5xx, gateway-error, connect-failure, retriable-4xx, reset"),
                );
            }
        }

        if policy.per_try_timeout_ms == 0 {
            issues.push(ValidationIssue::error(
                "RetryPolicy",
                resource_name,
                namespace,
                "perTryTimeout",
                "perTryTimeout is 0; retries will fail immediately",
            ));
        }

        if policy.backoff_base_ms > policy.backoff_max_ms && policy.backoff_max_ms > 0 {
            issues.push(ValidationIssue::warning(
                "RetryPolicy",
                resource_name,
                namespace,
                "retryBackOff",
                format!(
                    "backoffBase {}ms > backoffMax {}ms",
                    policy.backoff_base_ms, policy.backoff_max_ms
                ),
            ));
        }

        issues
    }

    // ------------------------------------------------------------------
    // Timeout consistency
    // ------------------------------------------------------------------

    pub fn check_timeout_consistency(
        &self,
        upstream_timeout_ms: u64,
        downstream_timeout_ms: u64,
        upstream_name: &str,
        downstream_name: &str,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if upstream_timeout_ms > 0
            && downstream_timeout_ms > 0
            && upstream_timeout_ms < downstream_timeout_ms
        {
            issues.push(
                ValidationIssue::error(
                    "TimeoutPolicy",
                    upstream_name,
                    "",
                    "requestTimeout",
                    format!(
                        "Upstream timeout ({}ms) < downstream timeout ({}ms) for '{}'. \
                         The upstream will time out before the downstream finishes.",
                        upstream_timeout_ms, downstream_timeout_ms, downstream_name
                    ),
                )
                .with_suggestion(&format!(
                    "Set upstream timeout >= downstream timeout (at least {}ms)",
                    downstream_timeout_ms
                )),
            );
        }

        if upstream_timeout_ms > self.config.max_timeout_ms {
            issues.push(
                ValidationIssue::warning(
                    "TimeoutPolicy",
                    upstream_name,
                    "",
                    "requestTimeout",
                    format!(
                        "Timeout {}ms exceeds recommended maximum {}ms",
                        upstream_timeout_ms, self.config.max_timeout_ms
                    ),
                )
                .with_suggestion(&format!(
                    "Reduce timeout to at most {}ms",
                    self.config.max_timeout_ms
                )),
            );
        }

        if upstream_timeout_ms > 0 && upstream_timeout_ms < self.config.min_timeout_ms {
            issues.push(
                ValidationIssue::warning(
                    "TimeoutPolicy",
                    upstream_name,
                    "",
                    "requestTimeout",
                    format!(
                        "Timeout {}ms is below recommended minimum {}ms",
                        upstream_timeout_ms, self.config.min_timeout_ms
                    ),
                )
                .with_suggestion(&format!(
                    "Increase timeout to at least {}ms",
                    self.config.min_timeout_ms
                )),
            );
        }

        issues
    }

    // ------------------------------------------------------------------
    // Missing policies
    // ------------------------------------------------------------------

    pub fn check_missing_policies(&self, resolved: &ResolvedRefs) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for (svc_id, policies) in &resolved.policy_map {
            if self.config.require_retry_policy && policies.retry_policy.is_none() {
                issues.push(
                    ValidationIssue::info(
                        "Service",
                        &svc_id.name,
                        &svc_id.namespace,
                        "retryPolicy",
                        format!("Service '{}' has no retry policy configured", svc_id),
                    )
                    .with_suggestion(
                        "Add a retry policy via Istio VirtualService or Envoy route config",
                    ),
                );
            }
            if self.config.require_timeout_policy && policies.timeout_policy.is_none() {
                issues.push(
                    ValidationIssue::info(
                        "Service",
                        &svc_id.name,
                        &svc_id.namespace,
                        "timeoutPolicy",
                        format!("Service '{}' has no timeout policy configured", svc_id),
                    )
                    .with_suggestion(
                        "Add a timeout via Istio VirtualService or Envoy route config",
                    ),
                );
            }
        }

        // Also flag services in the map that have no entry in policy_map at all.
        for svc_id in resolved.service_map.keys() {
            if !resolved.policy_map.contains_key(svc_id) {
                if self.config.require_retry_policy {
                    issues.push(
                        ValidationIssue::info(
                            "Service",
                            &svc_id.name,
                            &svc_id.namespace,
                            "retryPolicy",
                            format!(
                                "Service '{}' is known but has no policy entry at all",
                                svc_id
                            ),
                        )
                        .with_suggestion("Configure a retry and timeout policy for this service"),
                    );
                }
            }
        }

        issues
    }

    // ------------------------------------------------------------------
    // Circular dependency detection
    // ------------------------------------------------------------------

    pub fn check_circular_dependencies(
        &self,
        deps: &[(ServiceId, ServiceId)],
    ) -> Vec<ValidationIssue> {
        let cycles = find_cycles(deps);
        let mut issues = Vec::new();

        for cycle in &cycles {
            let path = cycle
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(" -> ");
            issues.push(
                ValidationIssue::critical(
                    "ServiceTopology",
                    &cycle[0].name,
                    &cycle[0].namespace,
                    "dependency_graph",
                    format!("Circular dependency detected: {}", path),
                )
                .with_suggestion("Break the dependency cycle to prevent cascading failures"),
            );
        }

        if !cycles.is_empty() {
            // Also check depth.
            let max_depth = compute_max_depth(deps);
            if max_depth > self.config.max_cascade_depth {
                issues.push(
                    ValidationIssue::warning(
                        "ServiceTopology",
                        "topology",
                        "",
                        "depth",
                        format!(
                            "Maximum cascade depth {} exceeds limit {}",
                            max_depth, self.config.max_cascade_depth
                        ),
                    )
                    .with_suggestion("Reduce service chain depth to limit retry amplification"),
                );
            }
        }

        // Even without cycles, check depth.
        let max_depth = compute_max_depth(deps);
        if max_depth > self.config.max_cascade_depth {
            issues.push(
                ValidationIssue::warning(
                    "ServiceTopology",
                    "topology",
                    "",
                    "depth",
                    format!(
                        "Maximum cascade depth {} exceeds limit {}",
                        max_depth, self.config.max_cascade_depth
                    ),
                )
                .with_suggestion("Reduce service chain depth to limit retry amplification"),
            );
        }

        issues
    }

    // ------------------------------------------------------------------
    // Orphaned resources
    // ------------------------------------------------------------------

    pub fn check_orphaned_resources(
        &self,
        k8s: &[KubernetesResource],
        istio: &[IstioConfig],
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Collect all hosts referenced in Istio configs.
        let mut referenced_hosts: HashSet<String> = HashSet::new();
        for cfg in istio {
            match cfg {
                IstioConfig::VirtualService(vs) => {
                    for route in &vs.http_routes {
                        for dest in &route.route {
                            let short = dest.destination.host.split('.').next().unwrap_or("");
                            referenced_hosts.insert(short.to_string());
                        }
                    }
                }
                IstioConfig::DestinationRule(dr) => {
                    let short = dr.host.split('.').next().unwrap_or("");
                    referenced_hosts.insert(short.to_string());
                }
                _ => {}
            }
        }

        // Identify k8s services that no Istio config references.
        for res in k8s {
            if let KubernetesResource::Service(svc) = res {
                if !referenced_hosts.contains(&svc.metadata.name) && !istio.is_empty() {
                    issues.push(
                        ValidationIssue::info(
                            "Service",
                            &svc.metadata.name,
                            &svc.metadata.namespace,
                            "",
                            format!(
                                "Service '{}' is not referenced by any Istio VirtualService or DestinationRule",
                                svc.metadata.name
                            ),
                        )
                        .with_suggestion("Consider adding Istio configuration for this service if it participates in the mesh"),
                    );
                }
            }
        }

        // Identify deployments whose labels don't match any service selector.
        let svc_selectors: Vec<&IndexMap<String, String>> = k8s
            .iter()
            .filter_map(|r| {
                if let KubernetesResource::Service(s) = r {
                    if s.spec.selector.is_empty() {
                        None
                    } else {
                        Some(&s.spec.selector)
                    }
                } else {
                    None
                }
            })
            .collect();

        for res in k8s {
            if let KubernetesResource::Deployment(dep) = res {
                let pod_labels = &dep.spec.template.metadata.labels;
                let matched = svc_selectors.iter().any(|sel| {
                    sel.iter()
                        .all(|(k, v)| pod_labels.get(k).map_or(false, |pv| pv == v))
                });
                if !matched && !svc_selectors.is_empty() {
                    issues.push(
                        ValidationIssue::info(
                            "Deployment",
                            &dep.metadata.name,
                            &dep.metadata.namespace,
                            "spec.template.metadata.labels",
                            format!(
                                "Deployment '{}' is not selected by any Service",
                                dep.metadata.name
                            ),
                        )
                        .with_suggestion("Ensure the deployment's pod labels match a Service selector"),
                    );
                }
            }
        }

        issues
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

fn validate_container(
    container: &ContainerSpec,
    deploy_name: &str,
    namespace: &str,
    config: &ValidatorConfig,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Check resource limits.
    if config.require_resource_limits {
        if container.resources.cpu_limit.is_none() {
            issues.push(
                ValidationIssue::warning(
                    "Deployment",
                    deploy_name,
                    namespace,
                    &format!("spec.containers[{}].resources.limits.cpu", container.name),
                    format!("Container '{}' has no CPU limit", container.name),
                )
                .with_suggestion("Set a CPU limit to prevent resource starvation"),
            );
        }
        if container.resources.memory_limit.is_none() {
            issues.push(
                ValidationIssue::warning(
                    "Deployment",
                    deploy_name,
                    namespace,
                    &format!(
                        "spec.containers[{}].resources.limits.memory",
                        container.name
                    ),
                    format!("Container '{}' has no memory limit", container.name),
                )
                .with_suggestion("Set a memory limit to prevent OOM kills"),
            );
        }
    }

    // Check image tag.
    let image = &container.image;
    if image.ends_with(":latest") || (!image.contains(':') && !image.contains('@')) {
        issues.push(
            ValidationIssue::warning(
                "Deployment",
                deploy_name,
                namespace,
                &format!("spec.containers[{}].image", container.name),
                format!(
                    "Container '{}' uses image '{}' without an explicit version tag",
                    container.name, image
                ),
            )
            .with_suggestion("Pin the image to a specific version tag for reproducibility"),
        );
    }

    // Check probes.
    if container.probes.readiness.is_none() && container.probes.liveness.is_none() {
        issues.push(
            ValidationIssue::info(
                "Deployment",
                deploy_name,
                namespace,
                &format!("spec.containers[{}].probes", container.name),
                format!("Container '{}' has no health probes configured", container.name),
            )
            .with_suggestion("Add at least a readiness probe for proper traffic management"),
        );
    }

    issues
}

fn validate_http_route(
    route: &HttpRoute,
    vs_name: &str,
    namespace: &str,
    config: &ValidatorConfig,
    route_index: usize,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let field_prefix = format!("spec.http[{}]", route_index);

    // Check route weights sum to 100 (if weights are used).
    if route.route.len() > 1 {
        let total_weight: u32 = route.route.iter().map(|r| r.weight).sum();
        if total_weight != 100 && total_weight != 0 {
            issues.push(
                ValidationIssue::error(
                    "VirtualService",
                    vs_name,
                    namespace,
                    &format!("{}.route[].weight", field_prefix),
                    format!(
                        "Route weights sum to {} instead of 100",
                        total_weight
                    ),
                )
                .with_suggestion("Adjust route weights to sum to exactly 100"),
            );
        }
    }

    // Validate retry policy.
    if let Some(ref retry) = route.retries {
        if retry.attempts > config.max_retry_attempts {
            issues.push(
                ValidationIssue::warning(
                    "VirtualService",
                    vs_name,
                    namespace,
                    &format!("{}.retries.attempts", field_prefix),
                    format!(
                        "Retry attempts {} exceeds recommended max {}",
                        retry.attempts, config.max_retry_attempts
                    ),
                )
                .with_suggestion(&format!(
                    "Reduce retry attempts to at most {}",
                    config.max_retry_attempts
                )),
            );
        }
        if retry.attempts < config.min_retry_attempts {
            issues.push(ValidationIssue::info(
                "VirtualService",
                vs_name,
                namespace,
                &format!("{}.retries.attempts", field_prefix),
                format!(
                    "Retry attempts {} is below recommended min {}",
                    retry.attempts, config.min_retry_attempts
                ),
            ));
        }
    }

    // Validate timeout.
    if let Some(ref timeout_str) = route.timeout {
        if let Ok(ms) = parse_simple_duration(timeout_str) {
            if ms > config.max_timeout_ms {
                issues.push(
                    ValidationIssue::warning(
                        "VirtualService",
                        vs_name,
                        namespace,
                        &format!("{}.timeout", field_prefix),
                        format!(
                            "Timeout {}ms exceeds recommended max {}ms",
                            ms, config.max_timeout_ms
                        ),
                    )
                    .with_suggestion(&format!(
                        "Reduce timeout to at most {}ms",
                        config.max_timeout_ms
                    )),
                );
            }
            if ms > 0 && ms < config.min_timeout_ms {
                issues.push(
                    ValidationIssue::warning(
                        "VirtualService",
                        vs_name,
                        namespace,
                        &format!("{}.timeout", field_prefix),
                        format!(
                            "Timeout {}ms is below recommended min {}ms",
                            ms, config.min_timeout_ms
                        ),
                    )
                    .with_suggestion(&format!(
                        "Increase timeout to at least {}ms",
                        config.min_timeout_ms
                    )),
                );
            }
        }
    }

    issues
}

/// Parse a simple duration string like "5s", "100ms", "1m" into milliseconds.
fn parse_simple_duration(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("ms") {
        rest.trim()
            .parse::<u64>()
            .map_err(|e| anyhow::anyhow!("invalid ms duration: {}", e))
    } else if let Some(rest) = s.strip_suffix('s') {
        let secs: f64 = rest
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid seconds duration: {}", e))?;
        Ok((secs * 1000.0) as u64)
    } else if let Some(rest) = s.strip_suffix('m') {
        let mins: f64 = rest
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid minutes duration: {}", e))?;
        Ok((mins * 60_000.0) as u64)
    } else {
        s.parse::<u64>()
            .map_err(|e| anyhow::anyhow!("cannot parse duration '{}': {}", s, e))
    }
}

/// Find all cycles in a directed graph represented as edge pairs.
fn find_cycles(edges: &[(ServiceId, ServiceId)]) -> Vec<Vec<ServiceId>> {
    let mut adjacency: HashMap<&ServiceId, Vec<&ServiceId>> = HashMap::new();
    let mut all_nodes: HashSet<&ServiceId> = HashSet::new();

    for (from, to) in edges {
        adjacency.entry(from).or_default().push(to);
        all_nodes.insert(from);
        all_nodes.insert(to);
    }

    let mut visited: HashSet<&ServiceId> = HashSet::new();
    let mut on_stack: HashSet<&ServiceId> = HashSet::new();
    let mut path: Vec<&ServiceId> = Vec::new();
    let mut cycles: Vec<Vec<ServiceId>> = Vec::new();

    fn dfs<'a>(
        node: &'a ServiceId,
        adjacency: &HashMap<&'a ServiceId, Vec<&'a ServiceId>>,
        visited: &mut HashSet<&'a ServiceId>,
        on_stack: &mut HashSet<&'a ServiceId>,
        path: &mut Vec<&'a ServiceId>,
        cycles: &mut Vec<Vec<ServiceId>>,
    ) {
        visited.insert(node);
        on_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = adjacency.get(node) {
            for &next in neighbors {
                if !visited.contains(next) {
                    dfs(next, adjacency, visited, on_stack, path, cycles);
                } else if on_stack.contains(next) {
                    // Found cycle — extract it from the path.
                    let start_idx = path.iter().position(|n| *n == next).unwrap_or(0);
                    let cycle: Vec<ServiceId> = path[start_idx..]
                        .iter()
                        .map(|s| (*s).clone())
                        .collect();
                    if cycle.len() >= 2 {
                        cycles.push(cycle);
                    }
                }
            }
        }

        path.pop();
        on_stack.remove(node);
    }

    for node in &all_nodes {
        if !visited.contains(*node) {
            dfs(node, &adjacency, &mut visited, &mut on_stack, &mut path, &mut cycles);
        }
    }

    cycles
}

/// Compute the maximum depth of any path in the dependency graph.
fn compute_max_depth(edges: &[(ServiceId, ServiceId)]) -> usize {
    if edges.is_empty() {
        return 0;
    }

    let mut adjacency: HashMap<&ServiceId, Vec<&ServiceId>> = HashMap::new();
    let mut all_nodes: HashSet<&ServiceId> = HashSet::new();
    let mut has_incoming: HashSet<&ServiceId> = HashSet::new();

    for (from, to) in edges {
        adjacency.entry(from).or_default().push(to);
        all_nodes.insert(from);
        all_nodes.insert(to);
        has_incoming.insert(to);
    }

    // Start BFS from roots (nodes with no incoming edges).
    let roots: Vec<&ServiceId> = all_nodes
        .iter()
        .filter(|n| !has_incoming.contains(*n))
        .copied()
        .collect();

    if roots.is_empty() {
        // All nodes are in cycles — return edge count as proxy.
        return edges.len();
    }

    let mut max_depth: usize = 0;
    for root in roots {
        let mut queue: VecDeque<(&ServiceId, usize)> = VecDeque::new();
        let mut visited: HashSet<&ServiceId> = HashSet::new();
        queue.push_back((root, 0));
        visited.insert(root);

        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            if let Some(neighbors) = adjacency.get(node) {
                for &next in neighbors {
                    if visited.insert(next) {
                        queue.push_back((next, depth + 1));
                    }
                }
            }
        }
    }

    max_depth
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::istio::{
        Destination, FaultAbort, FaultDelay, FaultInjection, HttpRetryPolicy, HttpRoute,
        HttpRouteDestination, OutlierDetection, Subset, TrafficPolicy,
    };
    use crate::kubernetes::{
        ContainerProbes, ContainerSpec, Deployment, DeploymentSpec, DeploymentStrategy,
        Ingress, KubeService, LabelSelector, PodTemplateSpec, ResourceRequirements,
        ServicePort, ServiceSpec,
    };
    use crate::reference_resolver::{ResolvedPolicies, ResolvedRefs, ResolvedService};
    use crate::ObjectMeta;
    use indexmap::IndexMap;

    fn make_deployment(
        name: &str,
        ns: &str,
        image: &str,
        labels: IndexMap<String, String>,
        has_limits: bool,
        has_probes: bool,
    ) -> Deployment {
        Deployment {
            metadata: ObjectMeta {
                name: name.into(),
                namespace: ns.into(),
                ..Default::default()
            },
            spec: DeploymentSpec {
                replicas: 2,
                selector: LabelSelector {
                    match_labels: labels.clone(),
                    match_expressions: vec![],
                },
                template: PodTemplateSpec {
                    metadata: ObjectMeta {
                        labels,
                        ..Default::default()
                    },
                    containers: vec![ContainerSpec {
                        name: "main".into(),
                        image: image.into(),
                        ports: vec![],
                        resources: if has_limits {
                            ResourceRequirements {
                                cpu_request: Some("100m".into()),
                                cpu_limit: Some("500m".into()),
                                memory_request: Some("128Mi".into()),
                                memory_limit: Some("256Mi".into()),
                            }
                        } else {
                            ResourceRequirements::default()
                        },
                        env: vec![],
                        probes: if has_probes {
                            ContainerProbes {
                                liveness: Some(crate::kubernetes::Probe {
                                    http_get: Some(crate::kubernetes::HttpGetAction {
                                        path: "/healthz".into(),
                                        port: 8080,
                                    }),
                                    tcp_socket: None,
                                    initial_delay_seconds: 5,
                                    period_seconds: 10,
                                    timeout_seconds: 3,
                                    failure_threshold: 3,
                                }),
                                readiness: None,
                                startup: None,
                            }
                        } else {
                            ContainerProbes::default()
                        },
                        command: vec![],
                        args: vec![],
                        volume_mounts: vec![],
                    }],
                    init_containers: vec![],
                    volumes: vec![],
                    service_account: None,
                },
                strategy: DeploymentStrategy {
                    strategy_type: "RollingUpdate".into(),
                    max_unavailable: None,
                    max_surge: None,
                },
            },
        }
    }

    fn make_service(
        name: &str,
        ns: &str,
        selector: IndexMap<String, String>,
    ) -> KubeService {
        KubeService {
            metadata: ObjectMeta {
                name: name.into(),
                namespace: ns.into(),
                ..Default::default()
            },
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

    fn labels(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_validate_deployment_missing_resources() {
        let v = ConfigValidator::with_defaults();
        let dep = make_deployment("api", "default", "api:v1", labels(&[("app", "api")]), false, true);
        let issues = v.validate_kubernetes_config(&[KubernetesResource::Deployment(dep)]);
        assert!(
            issues.iter().any(|i| i.message.contains("CPU limit")),
            "Should flag missing CPU limit"
        );
        assert!(
            issues.iter().any(|i| i.message.contains("memory limit")),
            "Should flag missing memory limit"
        );
    }

    #[test]
    fn test_validate_deployment_latest_tag() {
        let v = ConfigValidator::with_defaults();
        let dep = make_deployment("api", "default", "api:latest", labels(&[("app", "api")]), true, true);
        let issues = v.validate_kubernetes_config(&[KubernetesResource::Deployment(dep)]);
        assert!(
            issues.iter().any(|i| i.message.contains("version tag")),
            "Should flag :latest image"
        );
    }

    #[test]
    fn test_validate_deployment_no_probes() {
        let v = ConfigValidator::with_defaults();
        let dep = make_deployment("api", "default", "api:v1.2.3", labels(&[("app", "api")]), true, false);
        let issues = v.validate_kubernetes_config(&[KubernetesResource::Deployment(dep)]);
        assert!(
            issues.iter().any(|i| i.message.contains("health probes")),
            "Should flag missing probes"
        );
    }

    #[test]
    fn test_validate_service_no_matching_deployment() {
        let v = ConfigValidator::with_defaults();
        let svc = make_service("api-svc", "default", labels(&[("app", "api")]));
        let dep = make_deployment("worker", "default", "worker:v1", labels(&[("app", "worker")]), true, true);
        let issues = v.validate_kubernetes_config(&[
            KubernetesResource::Service(svc),
            KubernetesResource::Deployment(dep),
        ]);
        assert!(
            issues.iter().any(|i| i.message.contains("No deployment matches")),
            "Should flag unmatched selector"
        );
    }

    #[test]
    fn test_validate_retry_too_high() {
        let v = ConfigValidator::with_defaults();
        let policy = RetryPolicy {
            max_retries: 10,
            per_try_timeout_ms: 1000,
            retry_on: vec!["5xx".into()],
            backoff_base_ms: 25,
            backoff_max_ms: 250,
        };
        let issues = v.check_retry_bounds(&policy, "my-svc", "default");
        assert!(
            issues.iter().any(|i| i.message.contains("exceeds")),
            "Should flag high retry count"
        );
    }

    #[test]
    fn test_validate_retry_too_low() {
        let v = ConfigValidator::new(ValidatorConfig {
            min_retry_attempts: 2,
            ..Default::default()
        });
        let policy = RetryPolicy {
            max_retries: 1,
            per_try_timeout_ms: 1000,
            retry_on: vec!["5xx".into()],
            backoff_base_ms: 25,
            backoff_max_ms: 250,
        };
        let issues = v.check_retry_bounds(&policy, "my-svc", "default");
        assert!(
            issues.iter().any(|i| i.message.contains("below")),
            "Should flag low retry count"
        );
    }

    #[test]
    fn test_validate_timeout_consistency() {
        let v = ConfigValidator::with_defaults();
        // Upstream 2s < downstream 5s — problematic.
        let issues = v.check_timeout_consistency(2000, 5000, "gateway", "backend");
        assert!(
            issues.iter().any(|i| i.message.contains("Upstream timeout")),
            "Should flag upstream < downstream"
        );
    }

    #[test]
    fn test_validate_route_weights() {
        let v = ConfigValidator::with_defaults();
        let vs = VirtualService {
            metadata: ObjectMeta {
                name: "my-vs".into(),
                namespace: "default".into(),
                ..Default::default()
            },
            hosts: vec!["my-svc".into()],
            gateways: vec![],
            http_routes: vec![HttpRoute {
                name: None,
                match_conditions: vec![],
                route: vec![
                    HttpRouteDestination {
                        destination: Destination {
                            host: "my-svc".into(),
                            port: None,
                            subset: Some("v1".into()),
                        },
                        weight: 60,
                        headers: None,
                    },
                    HttpRouteDestination {
                        destination: Destination {
                            host: "my-svc".into(),
                            port: None,
                            subset: Some("v2".into()),
                        },
                        weight: 30,
                        headers: None,
                    },
                ],
                retries: None,
                timeout: None,
                fault: None,
                mirror: None,
                headers: None,
                rewrite: None,
            }],
            tcp_routes: vec![],
            tls_routes: vec![],
            export_to: vec![],
        };
        let issues = v.validate_istio_config(&[IstioConfig::VirtualService(vs)]);
        assert!(
            issues.iter().any(|i| i.message.contains("weights sum to")),
            "Should flag weights not summing to 100: {:?}",
            issues
        );
    }

    #[test]
    fn test_validate_outlier_detection() {
        let v = ConfigValidator::with_defaults();
        let dr = DestinationRule {
            metadata: ObjectMeta {
                name: "my-dr".into(),
                namespace: "default".into(),
                ..Default::default()
            },
            host: "my-svc".into(),
            traffic_policy: Some(TrafficPolicy {
                connection_pool: None,
                load_balancer: None,
                outlier_detection: Some(OutlierDetection {
                    consecutive_errors: 5,
                    interval: "10s".into(),
                    base_ejection_time: "30s".into(),
                    max_ejection_percent: 80,
                    min_health_percent: 50,
                }),
                tls: None,
                port_level_settings: vec![],
            }),
            subsets: vec![],
            export_to: vec![],
        };
        let issues = v.validate_istio_config(&[IstioConfig::DestinationRule(dr)]);
        assert!(
            issues.iter().any(|i| i.message.contains("ejection percent")),
            "Should flag high ejection percent: {:?}",
            issues
        );
    }

    #[test]
    fn test_validate_missing_policies() {
        let v = ConfigValidator::with_defaults();
        let sid = ServiceId::new("my-svc", "default");
        let resolved = ResolvedRefs {
            service_map: IndexMap::from([(
                sid.clone(),
                ResolvedService {
                    service: sid.clone(),
                    kubernetes_service: Some("my-svc".into()),
                    deployments: vec!["my-dep".into()],
                    virtual_services: vec![],
                    destination_rules: vec![],
                },
            )]),
            endpoint_map: IndexMap::new(),
            policy_map: IndexMap::from([(
                sid.clone(),
                ResolvedPolicies {
                    retry_policy: None,
                    timeout_policy: None,
                    source: crate::reference_resolver::PolicySource::Default,
                },
            )]),
            dependency_graph: vec![],
            unresolved: vec![],
        };
        let issues = v.check_missing_policies(&resolved);
        assert!(
            issues.iter().any(|i| i.message.contains("no retry")),
            "Should flag missing retry policy: {:?}",
            issues
        );
        assert!(
            issues.iter().any(|i| i.message.contains("no timeout")),
            "Should flag missing timeout policy: {:?}",
            issues
        );
    }

    #[test]
    fn test_check_circular_dependencies() {
        let v = ConfigValidator::with_defaults();
        let a = ServiceId::new("a", "default");
        let b = ServiceId::new("b", "default");
        let c = ServiceId::new("c", "default");
        let deps = vec![
            (a.clone(), b.clone()),
            (b.clone(), c.clone()),
            (c.clone(), a.clone()),
        ];
        let issues = v.check_circular_dependencies(&deps);
        assert!(
            issues.iter().any(|i| i.message.contains("Circular")),
            "Should detect cycle: {:?}",
            issues
        );
    }

    #[test]
    fn test_check_orphaned_resources() {
        let v = ConfigValidator::with_defaults();
        let svc = make_service("orphan", "default", labels(&[("app", "orphan")]));
        let dep = make_deployment("orphan-dep", "default", "orphan:v1", labels(&[("app", "orphan")]), true, true);
        let vs = VirtualService {
            metadata: ObjectMeta {
                name: "other-vs".into(),
                namespace: "default".into(),
                ..Default::default()
            },
            hosts: vec!["other-svc".into()],
            gateways: vec![],
            http_routes: vec![HttpRoute {
                name: None,
                match_conditions: vec![],
                route: vec![HttpRouteDestination {
                    destination: Destination {
                        host: "other-svc".into(),
                        port: None,
                        subset: None,
                    },
                    weight: 100,
                    headers: None,
                }],
                retries: None,
                timeout: None,
                fault: None,
                mirror: None,
                headers: None,
                rewrite: None,
            }],
            tcp_routes: vec![],
            tls_routes: vec![],
            export_to: vec![],
        };
        let k8s = vec![
            KubernetesResource::Service(svc),
            KubernetesResource::Deployment(dep),
        ];
        let istio = vec![IstioConfig::VirtualService(vs)];
        let issues = v.check_orphaned_resources(&k8s, &istio);
        assert!(
            issues.iter().any(|i| i.message.contains("orphan") && i.message.contains("not referenced")),
            "Should flag orphaned service: {:?}",
            issues
        );
    }

    #[test]
    fn test_validate_duplicate_virtual_service() {
        let v = ConfigValidator::with_defaults();
        let make_vs = |name: &str| VirtualService {
            metadata: ObjectMeta {
                name: name.into(),
                namespace: "default".into(),
                ..Default::default()
            },
            hosts: vec!["shared-host".into()],
            gateways: vec![],
            http_routes: vec![],
            tcp_routes: vec![],
            tls_routes: vec![],
            export_to: vec![],
        };
        let issues = v.validate_istio_config(&[
            IstioConfig::VirtualService(make_vs("vs-a")),
            IstioConfig::VirtualService(make_vs("vs-b")),
        ]);
        assert!(
            issues.iter().any(|i| i.message.contains("Multiple VirtualServices")),
            "Should detect duplicate VS hosts: {:?}",
            issues
        );
    }

    #[test]
    fn test_validate_empty_vs_hosts() {
        let v = ConfigValidator::with_defaults();
        let vs = VirtualService {
            metadata: ObjectMeta {
                name: "empty-vs".into(),
                namespace: "default".into(),
                ..Default::default()
            },
            hosts: vec![],
            gateways: vec![],
            http_routes: vec![],
            tcp_routes: vec![],
            tls_routes: vec![],
            export_to: vec![],
        };
        let issues = v.validate_istio_config(&[IstioConfig::VirtualService(vs)]);
        assert!(
            issues.iter().any(|i| i.message.contains("no hosts")),
            "Should flag empty hosts: {:?}",
            issues
        );
    }

    #[test]
    fn test_parse_simple_duration_variants() {
        assert_eq!(parse_simple_duration("5s").unwrap(), 5000);
        assert_eq!(parse_simple_duration("100ms").unwrap(), 100);
        assert_eq!(parse_simple_duration("1m").unwrap(), 60000);
        assert_eq!(parse_simple_duration("0.5s").unwrap(), 500);
    }

    #[test]
    fn test_no_issues_for_valid_config() {
        let v = ConfigValidator::with_defaults();
        let dep = make_deployment(
            "api",
            "default",
            "api:v1.2.3",
            labels(&[("app", "api")]),
            true,
            true,
        );
        let svc = make_service("api", "default", labels(&[("app", "api")]));
        let issues = v.validate_kubernetes_config(&[
            KubernetesResource::Deployment(dep),
            KubernetesResource::Service(svc),
        ]);
        // Should only have the "no probes" info at most for missing readiness.
        let errors: Vec<_> = issues.iter().filter(|i| i.severity >= Severity::Error).collect();
        assert!(errors.is_empty(), "Valid config should have no errors: {:?}", errors);
    }
}
