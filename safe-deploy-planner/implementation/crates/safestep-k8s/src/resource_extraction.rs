//! Resource extraction from Kubernetes manifests: services, versions, dependencies.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::manifest::{
    KubernetesManifest, ResourceQuantity,
};

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Service descriptor / Cluster resource model
// ---------------------------------------------------------------------------

/// Describes a service discovered from Kubernetes manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescriptor {
    pub name: String,
    pub namespace: String,
    pub kind: String,
    pub version: Option<ServiceVersion>,
    pub replicas: u32,
    pub containers: Vec<ContainerInfo>,
    pub labels: HashMap<String, String>,
    pub total_cpu_request: f64,
    pub total_memory_request: f64,
    pub total_cpu_limit: f64,
    pub total_memory_limit: f64,
    pub ports: Vec<u16>,
    pub dependencies: Vec<String>,
}

/// Extracted version information for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceVersion {
    pub raw: String,
    pub major: Option<u64>,
    pub minor: Option<u64>,
    pub patch: Option<u64>,
    pub pre_release: Option<String>,
    pub build: Option<String>,
}

impl std::fmt::Display for ServiceVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw)
    }
}

/// Container info extracted from a pod spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub name: String,
    pub image: String,
    pub registry: Option<String>,
    pub repository: String,
    pub tag: Option<String>,
    pub cpu_request: f64,
    pub cpu_limit: f64,
    pub memory_request: f64,
    pub memory_limit: f64,
}

/// A model of all resources in a cluster extracted from manifests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterResourceModel {
    pub services: Vec<ServiceDescriptor>,
    pub config_maps: Vec<String>,
    pub secrets: Vec<String>,
    pub persistent_volume_claims: Vec<String>,
    pub namespaces: Vec<String>,
    pub total_cpu_requests: f64,
    pub total_memory_requests: f64,
    pub total_cpu_limits: f64,
    pub total_memory_limits: f64,
    pub dependency_graph: HashMap<String, Vec<String>>,
}

// ---------------------------------------------------------------------------
// Resource extractor
// ---------------------------------------------------------------------------

/// Extracts structured resource information from a set of Kubernetes manifests.
pub struct ResourceExtractor;

impl ResourceExtractor {
    /// Extract service descriptors from workload manifests (Deployment, StatefulSet, DaemonSet).
    pub fn extract_services(manifests: &[KubernetesManifest]) -> Vec<ServiceDescriptor> {
        manifests
            .iter()
            .filter(|m| m.is_workload())
            .filter_map(|m| Self::extract_service_from_manifest(m))
            .collect()
    }

    /// Extract a full cluster resource model from all manifests.
    pub fn extract_resources(manifests: &[KubernetesManifest]) -> ClusterResourceModel {
        let services = Self::extract_services(manifests);
        let config_maps = manifests
            .iter()
            .filter(|m| m.kind == "ConfigMap")
            .map(|m| m.metadata.name.clone())
            .collect();
        let secrets = manifests
            .iter()
            .filter(|m| m.kind == "Secret")
            .map(|m| m.metadata.name.clone())
            .collect();
        let persistent_volume_claims = manifests
            .iter()
            .filter(|m| m.kind == "PersistentVolumeClaim")
            .map(|m| m.metadata.name.clone())
            .collect();
        let namespaces: Vec<String> = manifests
            .iter()
            .filter_map(|m| m.metadata.namespace.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let total_cpu_requests: f64 = services.iter().map(|s| s.total_cpu_request).sum();
        let total_memory_requests: f64 = services.iter().map(|s| s.total_memory_request).sum();
        let total_cpu_limits: f64 = services.iter().map(|s| s.total_cpu_limit).sum();
        let total_memory_limits: f64 = services.iter().map(|s| s.total_memory_limit).sum();

        let dependency_graph = DependencyExtractor::build_dependency_graph(&services);

        ClusterResourceModel {
            services,
            config_maps,
            secrets,
            persistent_volume_claims,
            namespaces,
            total_cpu_requests,
            total_memory_requests,
            total_cpu_limits,
            total_memory_limits,
            dependency_graph,
        }
    }

    fn extract_service_from_manifest(m: &KubernetesManifest) -> Option<ServiceDescriptor> {
        let spec = m.spec.as_ref()?;
        let replicas = spec.get("replicas").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let template_spec = spec
            .get("template")
            .and_then(|t| t.get("spec"));

        let containers_raw = template_spec
            .and_then(|s| s.get("containers"))
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        let containers: Vec<ContainerInfo> = containers_raw
            .iter()
            .map(|c| extract_container_info(c))
            .collect();

        // Version from the primary container image tag
        let version = containers
            .first()
            .and_then(|c| c.tag.as_ref())
            .and_then(|tag| VersionExtractor::from_tag(tag));

        let total_cpu_request: f64 =
            containers.iter().map(|c| c.cpu_request).sum::<f64>() * replicas as f64;
        let total_memory_request: f64 =
            containers.iter().map(|c| c.memory_request).sum::<f64>() * replicas as f64;
        let total_cpu_limit: f64 =
            containers.iter().map(|c| c.cpu_limit).sum::<f64>() * replicas as f64;
        let total_memory_limit: f64 =
            containers.iter().map(|c| c.memory_limit).sum::<f64>() * replicas as f64;

        let ports = extract_ports(template_spec);
        let dependencies = DependencyExtractor::from_env_vars(template_spec);
        let namespace = m.metadata.namespace.clone().unwrap_or_else(|| "default".to_string());

        Some(ServiceDescriptor {
            name: m.metadata.name.clone(),
            namespace,
            kind: m.kind.clone(),
            version,
            replicas,
            containers,
            labels: m.metadata.labels.clone(),
            total_cpu_request,
            total_memory_request,
            total_cpu_limit,
            total_memory_limit,
            ports,
            dependencies,
        })
    }
}

fn extract_container_info(c: &Value) -> ContainerInfo {
    let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let image_str = c.get("image").and_then(|i| i.as_str()).unwrap_or("").to_string();

    let (registry, repository, tag) = parse_image_ref(&image_str);

    let resources = c.get("resources");
    let cpu_request = resources
        .and_then(|r| r.get("requests"))
        .and_then(|r| r.get("cpu"))
        .and_then(|v| v.as_str())
        .and_then(|s| ResourceQuantity::parse_cpu(s).ok())
        .map(|q| q.value)
        .unwrap_or(0.0);
    let cpu_limit = resources
        .and_then(|r| r.get("limits"))
        .and_then(|r| r.get("cpu"))
        .and_then(|v| v.as_str())
        .and_then(|s| ResourceQuantity::parse_cpu(s).ok())
        .map(|q| q.value)
        .unwrap_or(0.0);
    let memory_request = resources
        .and_then(|r| r.get("requests"))
        .and_then(|r| r.get("memory"))
        .and_then(|v| v.as_str())
        .and_then(|s| ResourceQuantity::parse_memory(s).ok())
        .map(|q| q.value)
        .unwrap_or(0.0);
    let memory_limit = resources
        .and_then(|r| r.get("limits"))
        .and_then(|r| r.get("memory"))
        .and_then(|v| v.as_str())
        .and_then(|s| ResourceQuantity::parse_memory(s).ok())
        .map(|q| q.value)
        .unwrap_or(0.0);

    ContainerInfo {
        name,
        image: image_str,
        registry,
        repository,
        tag,
        cpu_request,
        cpu_limit,
        memory_request,
        memory_limit,
    }
}

fn parse_image_ref(image: &str) -> (Option<String>, String, Option<String>) {
    // Handle digest
    let image_no_digest = image.split('@').next().unwrap_or(image);

    // Split tag
    let (name_part, tag) = if let Some(colon_pos) = image_no_digest.rfind(':') {
        // Check if the colon is part of a port number (registry:port/repo)
        let after_colon = &image_no_digest[colon_pos + 1..];
        if after_colon.contains('/') {
            // It's a port, not a tag
            (image_no_digest, None)
        } else {
            (
                &image_no_digest[..colon_pos],
                Some(after_colon.to_string()),
            )
        }
    } else {
        (image_no_digest, None)
    };

    // Split registry/repository
    let parts: Vec<&str> = name_part.splitn(2, '/').collect();
    let (registry, repository) = if parts.len() == 2 && (parts[0].contains('.') || parts[0].contains(':') || parts[0] == "localhost") {
        (Some(parts[0].to_string()), parts[1].to_string())
    } else {
        (None, name_part.to_string())
    };

    (registry, repository, tag)
}

fn extract_ports(template_spec: Option<&Value>) -> Vec<u16> {
    template_spec
        .and_then(|s| s.get("containers"))
        .and_then(|c| c.as_array())
        .map(|containers| {
            containers
                .iter()
                .flat_map(|c| {
                    c.get("ports")
                        .and_then(|p| p.as_array())
                        .map(|ports| {
                            ports
                                .iter()
                                .filter_map(|p| p.get("containerPort").and_then(|v| v.as_u64()).map(|v| v as u16))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Version extractor
// ---------------------------------------------------------------------------

/// Extracts semantic versions from container image tags.
pub struct VersionExtractor;

impl VersionExtractor {
    /// Extract a version from a full image reference (e.g. "nginx:1.21.0").
    pub fn from_image_tag(image: &str) -> Option<ServiceVersion> {
        let tag = image.split(':').nth(1)?;
        Self::from_tag(tag)
    }

    /// Extract a version from a tag string.
    pub fn from_tag(tag: &str) -> Option<ServiceVersion> {
        let tag = tag.trim().trim_start_matches('v');
        if tag.is_empty() {
            return None;
        }

        // Try semver: major.minor.patch[-pre][+build]
        if let Some(v) = Self::parse_semver(tag) {
            return Some(v);
        }

        // Try date-based: YYYYMMDD or YYYY.MM.DD or YYYY-MM-DD
        if let Some(v) = Self::parse_date_version(tag) {
            return Some(v);
        }

        // Try simple major.minor
        if let Some(v) = Self::parse_major_minor(tag) {
            return Some(v);
        }

        // Treat as raw version (could be a commit hash or custom tag)
        Some(ServiceVersion {
            raw: tag.to_string(),
            major: None,
            minor: None,
            patch: None,
            pre_release: None,
            build: None,
        })
    }

    fn parse_semver(tag: &str) -> Option<ServiceVersion> {
        let re = regex::Regex::new(
            r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9._-]+))?(?:\+([a-zA-Z0-9._-]+))?$"
        ).ok()?;
        let caps = re.captures(tag)?;
        Some(ServiceVersion {
            raw: tag.to_string(),
            major: caps.get(1).and_then(|m| m.as_str().parse().ok()),
            minor: caps.get(2).and_then(|m| m.as_str().parse().ok()),
            patch: caps.get(3).and_then(|m| m.as_str().parse().ok()),
            pre_release: caps.get(4).map(|m| m.as_str().to_string()),
            build: caps.get(5).map(|m| m.as_str().to_string()),
        })
    }

    fn parse_date_version(tag: &str) -> Option<ServiceVersion> {
        // YYYYMMDD
        let re = regex::Regex::new(r"^(\d{4})(\d{2})(\d{2})$").ok()?;
        if let Some(caps) = re.captures(tag) {
            let year: u64 = caps[1].parse().ok()?;
            let month: u64 = caps[2].parse().ok()?;
            let day: u64 = caps[3].parse().ok()?;
            if (2000..=2099).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day) {
                return Some(ServiceVersion {
                    raw: tag.to_string(),
                    major: Some(year),
                    minor: Some(month),
                    patch: Some(day),
                    pre_release: None,
                    build: None,
                });
            }
        }

        // YYYY.MM.DD or YYYY-MM-DD
        let re2 = regex::Regex::new(r"^(\d{4})[.\-](\d{1,2})[.\-](\d{1,2})$").ok()?;
        if let Some(caps) = re2.captures(tag) {
            return Some(ServiceVersion {
                raw: tag.to_string(),
                major: caps.get(1).and_then(|m| m.as_str().parse().ok()),
                minor: caps.get(2).and_then(|m| m.as_str().parse().ok()),
                patch: caps.get(3).and_then(|m| m.as_str().parse().ok()),
                pre_release: None,
                build: None,
            });
        }

        None
    }

    fn parse_major_minor(tag: &str) -> Option<ServiceVersion> {
        let re = regex::Regex::new(r"^(\d+)\.(\d+)$").ok()?;
        let caps = re.captures(tag)?;
        Some(ServiceVersion {
            raw: tag.to_string(),
            major: caps.get(1).and_then(|m| m.as_str().parse().ok()),
            minor: caps.get(2).and_then(|m| m.as_str().parse().ok()),
            patch: None,
            pre_release: None,
            build: None,
        })
    }

    /// Compare two versions for ordering. Returns Ordering.
    pub fn compare(a: &ServiceVersion, b: &ServiceVersion) -> std::cmp::Ordering {
        let cmp_field = |a: &Option<u64>, b: &Option<u64>| -> std::cmp::Ordering {
            match (a, b) {
                (Some(a), Some(b)) => a.cmp(b),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            }
        };
        cmp_field(&a.major, &b.major)
            .then(cmp_field(&a.minor, &b.minor))
            .then(cmp_field(&a.patch, &b.patch))
    }
}

// ---------------------------------------------------------------------------
// Dependency extractor
// ---------------------------------------------------------------------------

/// Infers service dependencies from environment variables and other config.
pub struct DependencyExtractor;

impl DependencyExtractor {
    /// Extract dependencies from environment variables in a pod spec.
    pub fn from_env_vars(template_spec: Option<&Value>) -> Vec<String> {
        let mut deps = Vec::new();
        let containers = template_spec
            .and_then(|s| s.get("containers"))
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let init_containers = template_spec
            .and_then(|s| s.get("initContainers"))
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        let all_containers: Vec<&Value> = containers.iter().chain(init_containers.iter()).collect();

        for c in all_containers {
            if let Some(env_arr) = c.get("env").and_then(|e| e.as_array()) {
                for env in env_arr {
                    if let Some(value) = env.get("value").and_then(|v| v.as_str()) {
                        // Look for K8s service DNS names: <service>.<namespace>.svc.cluster.local
                        let extracted = Self::extract_service_dns_refs(value);
                        deps.extend(extracted);

                        // Look for common env var patterns like DATABASE_HOST=postgres
                        if Self::is_service_ref_env_var(
                            env.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                        ) {
                            let service = value.split('.').next().unwrap_or(value);
                            if Self::looks_like_service_name(service) {
                                deps.push(service.to_string());
                            }
                        }
                    }
                }
            }

            // Also check init containers for dependency signals
            if let Some(cmd) = c.get("command").and_then(|c| c.as_array()) {
                for arg in cmd {
                    if let Some(s) = arg.as_str() {
                        deps.extend(Self::extract_service_dns_refs(s));
                    }
                }
            }
            if let Some(args) = c.get("args").and_then(|a| a.as_array()) {
                for arg in args {
                    if let Some(s) = arg.as_str() {
                        deps.extend(Self::extract_service_dns_refs(s));
                    }
                }
            }
        }

        // Deduplicate
        deps.sort();
        deps.dedup();
        deps
    }

    /// Build a dependency graph from all services.
    pub fn build_dependency_graph(services: &[ServiceDescriptor]) -> HashMap<String, Vec<String>> {
        let service_names: std::collections::HashSet<&str> =
            services.iter().map(|s| s.name.as_str()).collect();
        let mut graph = HashMap::new();
        for svc in services {
            let valid_deps: Vec<String> = svc
                .dependencies
                .iter()
                .filter(|d| service_names.contains(d.as_str()))
                .cloned()
                .collect();
            graph.insert(svc.name.clone(), valid_deps);
        }
        graph
    }

    /// Extract service DNS references from a string.
    fn extract_service_dns_refs(s: &str) -> Vec<String> {
        let mut refs = Vec::new();
        // Match: <name>.<namespace>.svc.cluster.local or <name>.svc
        let re = regex::Regex::new(
            r"([a-z][a-z0-9-]+)(?:\.[a-z][a-z0-9-]+)?\.svc(?:\.cluster\.local)?"
        )
        .unwrap();
        for cap in re.captures_iter(s) {
            if let Some(m) = cap.get(1) {
                refs.push(m.as_str().to_string());
            }
        }
        refs
    }

    fn is_service_ref_env_var(name: &str) -> bool {
        let patterns = [
            "_HOST", "_ADDR", "_URL", "_ENDPOINT", "_SERVER", "_SERVICE", "_URI",
        ];
        let upper = name.to_uppercase();
        patterns.iter().any(|p| upper.ends_with(p))
    }

    fn looks_like_service_name(s: &str) -> bool {
        if s.is_empty() || s.len() > 63 {
            return false;
        }
        // Must look like a DNS name
        let re = regex::Regex::new(r"^[a-z][a-z0-9-]*$").unwrap();
        re.is_match(s) && !s.starts_with('-') && !s.ends_with('-')
    }
}

// ---------------------------------------------------------------------------
// Resource aggregator
// ---------------------------------------------------------------------------

/// Aggregates resource requirements across pods and replicas.
pub struct ResourceAggregator;

impl ResourceAggregator {
    /// Compute total CPU request for a service, considering replicas.
    pub fn total_cpu(service: &ServiceDescriptor) -> f64 {
        service.total_cpu_request
    }

    /// Compute total memory request for a service, considering replicas.
    pub fn total_memory(service: &ServiceDescriptor) -> f64 {
        service.total_memory_request
    }

    /// Compute total CPU with replicas and optional overhead.
    pub fn total_cpu_with_overhead(service: &ServiceDescriptor, overhead_percent: f64) -> f64 {
        service.total_cpu_request * (1.0 + overhead_percent / 100.0)
    }

    /// Compute total memory with replicas and optional overhead.
    pub fn total_memory_with_overhead(service: &ServiceDescriptor, overhead_percent: f64) -> f64 {
        service.total_memory_request * (1.0 + overhead_percent / 100.0)
    }

    /// Compute per-replica CPU request.
    pub fn per_replica_cpu(service: &ServiceDescriptor) -> f64 {
        if service.replicas == 0 {
            return 0.0;
        }
        service.total_cpu_request / service.replicas as f64
    }

    /// Compute per-replica memory request.
    pub fn per_replica_memory(service: &ServiceDescriptor) -> f64 {
        if service.replicas == 0 {
            return 0.0;
        }
        service.total_memory_request / service.replicas as f64
    }

    /// Re-compute totals with a different replica count.
    pub fn with_replicas(service: &ServiceDescriptor, new_replicas: u32) -> (f64, f64) {
        let per_cpu = Self::per_replica_cpu(service);
        let per_mem = Self::per_replica_memory(service);
        (per_cpu * new_replicas as f64, per_mem * new_replicas as f64)
    }

    /// Aggregate across all services.
    pub fn aggregate(services: &[ServiceDescriptor]) -> (f64, f64, f64, f64) {
        let total_cpu_req = services.iter().map(|s| s.total_cpu_request).sum();
        let total_mem_req = services.iter().map(|s| s.total_memory_request).sum();
        let total_cpu_lim = services.iter().map(|s| s.total_cpu_limit).sum();
        let total_mem_lim = services.iter().map(|s| s.total_memory_limit).sum();
        (total_cpu_req, total_mem_req, total_cpu_lim, total_mem_lim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::KubernetesManifest;

    fn sample_deployment() -> KubernetesManifest {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-api
  namespace: production
  labels:
    app: web-api
    version: "1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-api
  template:
    metadata:
      labels:
        app: web-api
    spec:
      containers:
      - name: api
        image: registry.io/myorg/web-api:1.2.3
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        env:
        - name: DATABASE_HOST
          value: postgres.production.svc.cluster.local
        - name: REDIS_ADDR
          value: redis
        - name: LOG_LEVEL
          value: info
      initContainers:
      - name: wait-for-db
        image: busybox
        command: ["sh", "-c", "until nslookup postgres.production.svc.cluster.local; do sleep 2; done"]
"#;
        KubernetesManifest::parse(yaml).unwrap().into_iter().next().unwrap()
    }

    fn sample_statefulset() -> KubernetesManifest {
        let yaml = r#"
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: production
spec:
  replicas: 2
  serviceName: postgres-headless
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15.2
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
"#;
        KubernetesManifest::parse(yaml).unwrap().into_iter().next().unwrap()
    }

    #[test]
    fn test_extract_services() {
        let manifests = vec![sample_deployment(), sample_statefulset()];
        let services = ResourceExtractor::extract_services(&manifests);
        assert_eq!(services.len(), 2);

        let web_api = services.iter().find(|s| s.name == "web-api").unwrap();
        assert_eq!(web_api.namespace, "production");
        assert_eq!(web_api.kind, "Deployment");
        assert_eq!(web_api.replicas, 3);
        assert_eq!(web_api.ports, vec![8080]);
        assert!(web_api.version.is_some());
        let ver = web_api.version.as_ref().unwrap();
        assert_eq!(ver.major, Some(1));
        assert_eq!(ver.minor, Some(2));
        assert_eq!(ver.patch, Some(3));

        let pg = services.iter().find(|s| s.name == "postgres").unwrap();
        assert_eq!(pg.replicas, 2);
        assert_eq!(pg.kind, "StatefulSet");
    }

    #[test]
    fn test_extract_cluster_resource_model() {
        let manifests = vec![sample_deployment(), sample_statefulset()];
        let model = ResourceExtractor::extract_resources(&manifests);
        assert_eq!(model.services.len(), 2);
        assert!(model.total_cpu_requests > 0.0);
        assert!(model.total_memory_requests > 0.0);
        assert!(model.namespaces.contains(&"production".to_string()));
    }

    #[test]
    fn test_version_extractor_semver() {
        let v = VersionExtractor::from_image_tag("nginx:1.21.0").unwrap();
        assert_eq!(v.major, Some(1));
        assert_eq!(v.minor, Some(21));
        assert_eq!(v.patch, Some(0));

        let v2 = VersionExtractor::from_image_tag("app:v2.3.4-beta.1").unwrap();
        assert_eq!(v2.major, Some(2));
        assert_eq!(v2.minor, Some(3));
        assert_eq!(v2.patch, Some(4));
        assert_eq!(v2.pre_release.as_deref(), Some("beta.1"));
    }

    #[test]
    fn test_version_extractor_date() {
        let v = VersionExtractor::from_tag("20231215").unwrap();
        assert_eq!(v.major, Some(2023));
        assert_eq!(v.minor, Some(12));
        assert_eq!(v.patch, Some(15));

        let v2 = VersionExtractor::from_tag("2023.12.15").unwrap();
        assert_eq!(v2.major, Some(2023));
    }

    #[test]
    fn test_version_extractor_major_minor() {
        let v = VersionExtractor::from_tag("15.2").unwrap();
        assert_eq!(v.major, Some(15));
        assert_eq!(v.minor, Some(2));
        assert!(v.patch.is_none());
    }

    #[test]
    fn test_version_extractor_commit_hash() {
        let v = VersionExtractor::from_tag("abc123def").unwrap();
        assert_eq!(v.raw, "abc123def");
        assert!(v.major.is_none());
    }

    #[test]
    fn test_version_extractor_latest() {
        let v = VersionExtractor::from_tag("latest").unwrap();
        assert_eq!(v.raw, "latest");
        assert!(v.major.is_none());
    }

    #[test]
    fn test_version_comparison() {
        let v1 = VersionExtractor::from_tag("1.2.3").unwrap();
        let v2 = VersionExtractor::from_tag("1.2.4").unwrap();
        let v3 = VersionExtractor::from_tag("2.0.0").unwrap();
        assert_eq!(
            VersionExtractor::compare(&v1, &v2),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            VersionExtractor::compare(&v2, &v3),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            VersionExtractor::compare(&v1, &v1),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn test_dependency_extractor_env_vars() {
        let spec = serde_json::json!({
            "containers": [{
                "name": "app",
                "env": [
                    {"name": "DB_HOST", "value": "postgres.default.svc.cluster.local"},
                    {"name": "CACHE_ADDR", "value": "redis"},
                    {"name": "LOG_LEVEL", "value": "debug"}
                ]
            }]
        });
        let deps = DependencyExtractor::from_env_vars(Some(&spec));
        assert!(deps.contains(&"postgres".to_string()));
        assert!(deps.contains(&"redis".to_string()));
        assert!(!deps.contains(&"debug".to_string()));
    }

    #[test]
    fn test_dependency_extractor_init_containers() {
        let spec = serde_json::json!({
            "containers": [{"name": "app"}],
            "initContainers": [{
                "name": "wait",
                "command": ["sh", "-c", "until nslookup db.default.svc.cluster.local; do sleep 1; done"]
            }]
        });
        let deps = DependencyExtractor::from_env_vars(Some(&spec));
        assert!(deps.contains(&"db".to_string()));
    }

    #[test]
    fn test_dependency_graph() {
        let services = vec![
            ServiceDescriptor {
                name: "web".into(),
                namespace: "default".into(),
                kind: "Deployment".into(),
                version: None,
                replicas: 1,
                containers: Vec::new(),
                labels: HashMap::new(),
                total_cpu_request: 0.0,
                total_memory_request: 0.0,
                total_cpu_limit: 0.0,
                total_memory_limit: 0.0,
                ports: Vec::new(),
                dependencies: vec!["api".into(), "external".into()],
            },
            ServiceDescriptor {
                name: "api".into(),
                namespace: "default".into(),
                kind: "Deployment".into(),
                version: None,
                replicas: 1,
                containers: Vec::new(),
                labels: HashMap::new(),
                total_cpu_request: 0.0,
                total_memory_request: 0.0,
                total_cpu_limit: 0.0,
                total_memory_limit: 0.0,
                ports: Vec::new(),
                dependencies: vec!["db".into()],
            },
            ServiceDescriptor {
                name: "db".into(),
                namespace: "default".into(),
                kind: "StatefulSet".into(),
                version: None,
                replicas: 1,
                containers: Vec::new(),
                labels: HashMap::new(),
                total_cpu_request: 0.0,
                total_memory_request: 0.0,
                total_cpu_limit: 0.0,
                total_memory_limit: 0.0,
                ports: Vec::new(),
                dependencies: Vec::new(),
            },
        ];
        let graph = DependencyExtractor::build_dependency_graph(&services);
        assert_eq!(graph.get("web").unwrap(), &vec!["api".to_string()]);
        assert_eq!(graph.get("api").unwrap(), &vec!["db".to_string()]);
        assert!(graph.get("db").unwrap().is_empty());
    }

    #[test]
    fn test_resource_aggregator() {
        let svc = ServiceDescriptor {
            name: "test".into(),
            namespace: "default".into(),
            kind: "Deployment".into(),
            version: None,
            replicas: 3,
            containers: Vec::new(),
            labels: HashMap::new(),
            total_cpu_request: 0.75,   // 3 * 0.25
            total_memory_request: 384.0 * 1024.0 * 1024.0,  // 3 * 128Mi
            total_cpu_limit: 1.5,
            total_memory_limit: 768.0 * 1024.0 * 1024.0,
            ports: vec![8080],
            dependencies: Vec::new(),
        };

        assert!((ResourceAggregator::total_cpu(&svc) - 0.75).abs() < 0.001);
        assert!((ResourceAggregator::per_replica_cpu(&svc) - 0.25).abs() < 0.001);

        let (new_cpu, _new_mem) = ResourceAggregator::with_replicas(&svc, 5);
        assert!((new_cpu - 1.25).abs() < 0.001);

        let with_overhead = ResourceAggregator::total_cpu_with_overhead(&svc, 10.0);
        assert!((with_overhead - 0.825).abs() < 0.001);
    }

    #[test]
    fn test_resource_aggregator_aggregate() {
        let services = vec![
            ServiceDescriptor {
                name: "a".into(),
                namespace: "default".into(),
                kind: "Deployment".into(),
                version: None,
                replicas: 1,
                containers: Vec::new(),
                labels: HashMap::new(),
                total_cpu_request: 0.5,
                total_memory_request: 100.0,
                total_cpu_limit: 1.0,
                total_memory_limit: 200.0,
                ports: Vec::new(),
                dependencies: Vec::new(),
            },
            ServiceDescriptor {
                name: "b".into(),
                namespace: "default".into(),
                kind: "Deployment".into(),
                version: None,
                replicas: 1,
                containers: Vec::new(),
                labels: HashMap::new(),
                total_cpu_request: 0.3,
                total_memory_request: 50.0,
                total_cpu_limit: 0.6,
                total_memory_limit: 100.0,
                ports: Vec::new(),
                dependencies: Vec::new(),
            },
        ];
        let (cpu_req, mem_req, cpu_lim, mem_lim) = ResourceAggregator::aggregate(&services);
        assert!((cpu_req - 0.8).abs() < 0.001);
        assert!((mem_req - 150.0).abs() < 0.001);
        assert!((cpu_lim - 1.6).abs() < 0.001);
        assert!((mem_lim - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_image_ref() {
        let (reg, repo, tag) = parse_image_ref("registry.io/myorg/app:1.0");
        assert_eq!(reg.as_deref(), Some("registry.io"));
        assert_eq!(repo, "myorg/app");
        assert_eq!(tag.as_deref(), Some("1.0"));

        let (reg, repo, tag) = parse_image_ref("nginx:latest");
        assert!(reg.is_none());
        assert_eq!(repo, "nginx");
        assert_eq!(tag.as_deref(), Some("latest"));

        let (reg, repo, tag) = parse_image_ref("nginx");
        assert!(reg.is_none());
        assert_eq!(repo, "nginx");
        assert!(tag.is_none());

        let (reg, repo, tag) = parse_image_ref("localhost:5000/myapp:v1");
        assert_eq!(reg.as_deref(), Some("localhost:5000"));
        assert_eq!(repo, "myapp");
        assert_eq!(tag.as_deref(), Some("v1"));
    }

    #[test]
    fn test_container_info_extraction() {
        let manifests = vec![sample_deployment()];
        let services = ResourceExtractor::extract_services(&manifests);
        let svc = &services[0];
        assert_eq!(svc.containers.len(), 1);
        let c = &svc.containers[0];
        assert_eq!(c.name, "api");
        assert_eq!(c.registry.as_deref(), Some("registry.io"));
        assert_eq!(c.repository, "myorg/web-api");
        assert_eq!(c.tag.as_deref(), Some("1.2.3"));
        assert!((c.cpu_request - 0.25).abs() < 0.001);
        assert!((c.memory_request - 256.0 * 1024.0 * 1024.0).abs() < 1.0);
    }

    #[test]
    fn test_service_does_not_extract_from_non_workloads() {
        let yaml = r#"
apiVersion: v1
kind: Service
metadata:
  name: my-svc
spec:
  type: ClusterIP
  ports:
  - port: 80
"#;
        let manifests = KubernetesManifest::parse(yaml).unwrap();
        let services = ResourceExtractor::extract_services(&manifests);
        assert!(services.is_empty());
    }
}
