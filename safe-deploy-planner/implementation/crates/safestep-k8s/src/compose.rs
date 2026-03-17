//! Docker Compose file parsing, service extraction, and version graph conversion.
//!
//! This module parses Docker Compose YAML files (both v2 and v3 formats),
//! extracts service definitions with their images, versions, dependencies,
//! health checks, and environment variables, and converts them into SafeStep's
//! version-product graph format for deployment planning.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::resource_extraction::{
    ContainerInfo, ServiceDescriptor, ServiceVersion, VersionExtractor,
};

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Compose format version
// ---------------------------------------------------------------------------

/// The Docker Compose file format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComposeFormatVersion {
    /// Version 2.x (docker-compose v2 schema).
    V2,
    /// Version 3.x (docker-compose v3 / Docker Swarm schema).
    V3,
    /// No explicit version field — modern Compose Specification.
    Modern,
}

impl fmt::Display for ComposeFormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V2 => write!(f, "2"),
            Self::V3 => write!(f, "3"),
            Self::Modern => write!(f, "modern"),
        }
    }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

/// Health check configuration extracted from a Compose service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeHealthCheck {
    /// The health check command (e.g. `["CMD", "curl", "-f", "http://localhost/"]`).
    pub test: Vec<String>,
    /// Interval between checks, as a raw duration string (e.g. "30s").
    pub interval: Option<String>,
    /// Timeout for a single check.
    pub timeout: Option<String>,
    /// Number of consecutive failures before the container is unhealthy.
    pub retries: Option<u32>,
    /// Grace period before health checks begin.
    pub start_period: Option<String>,
    /// If `true`, health checking is disabled for this service.
    pub disable: bool,
}

impl Default for ComposeHealthCheck {
    fn default() -> Self {
        Self {
            test: Vec::new(),
            interval: None,
            timeout: None,
            retries: None,
            start_period: None,
            disable: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Dependency condition
// ---------------------------------------------------------------------------

/// The condition that must be met before a dependency is considered satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyCondition {
    /// The dependency service has started (default).
    ServiceStarted,
    /// The dependency service reports healthy via its health check.
    ServiceHealthy,
    /// The dependency service has completed successfully (exit 0).
    ServiceCompletedSuccessfully,
}

impl Default for DependencyCondition {
    fn default() -> Self {
        Self::ServiceStarted
    }
}

impl fmt::Display for DependencyCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServiceStarted => write!(f, "service_started"),
            Self::ServiceHealthy => write!(f, "service_healthy"),
            Self::ServiceCompletedSuccessfully => write!(f, "service_completed_successfully"),
        }
    }
}

/// A dependency on another Compose service with an optional condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeDependency {
    /// Name of the service this depends on.
    pub service: String,
    /// Condition that must hold before the dependency is satisfied.
    pub condition: DependencyCondition,
}

// ---------------------------------------------------------------------------
// Port mapping
// ---------------------------------------------------------------------------

/// A port mapping extracted from a Compose service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposePort {
    /// Host port (may be `None` for ephemeral mapping).
    pub host: Option<u16>,
    /// Container port.
    pub container: u16,
    /// Protocol (tcp or udp).
    pub protocol: String,
}

// ---------------------------------------------------------------------------
// Volume mount
// ---------------------------------------------------------------------------

/// A volume mount extracted from a Compose service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeVolume {
    /// Source path or named volume.
    pub source: String,
    /// Container mount path.
    pub target: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
}

// ---------------------------------------------------------------------------
// Resource limits
// ---------------------------------------------------------------------------

/// Deploy resource limits/reservations (v3 deploy block).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComposeResourceLimits {
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
    pub cpu_reservation: Option<String>,
    pub memory_reservation: Option<String>,
}

// ---------------------------------------------------------------------------
// Restart policy
// ---------------------------------------------------------------------------

/// Restart policy for a Compose service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestartPolicy {
    No,
    Always,
    OnFailure,
    UnlessStopped,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self::No
    }
}

impl RestartPolicy {
    fn parse(s: &str) -> Self {
        match s {
            "always" => Self::Always,
            "on-failure" => Self::OnFailure,
            "unless-stopped" => Self::UnlessStopped,
            _ => Self::No,
        }
    }
}

// ---------------------------------------------------------------------------
// Network configuration
// ---------------------------------------------------------------------------

/// A network reference from a Compose service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeNetworkRef {
    /// Network name.
    pub name: String,
    /// Optional aliases for this service on the network.
    pub aliases: Vec<String>,
}

// ---------------------------------------------------------------------------
// Compose service
// ---------------------------------------------------------------------------

/// A fully parsed Docker Compose service definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeService {
    /// Service name (the key under `services:`).
    pub name: String,
    /// Full image reference (e.g. `nginx:1.25.3`).
    pub image: Option<String>,
    /// Build context path, if the service uses a Dockerfile instead of an image.
    pub build_context: Option<String>,
    /// Extracted version information from the image tag.
    pub version: Option<ServiceVersion>,
    /// The container name override, if specified.
    pub container_name: Option<String>,
    /// Hostname override.
    pub hostname: Option<String>,
    /// Restart policy.
    pub restart: RestartPolicy,
    /// Dependencies on other services, with conditions.
    pub dependencies: Vec<ComposeDependency>,
    /// Health check configuration.
    pub health_check: Option<ComposeHealthCheck>,
    /// Environment variables (name → value).
    pub environment: HashMap<String, String>,
    /// Labels attached to the service.
    pub labels: HashMap<String, String>,
    /// Port mappings.
    pub ports: Vec<ComposePort>,
    /// Volume mounts.
    pub volumes: Vec<ComposeVolume>,
    /// Network references.
    pub networks: Vec<ComposeNetworkRef>,
    /// Resource limits (from v3 `deploy` block or v2 resource keys).
    pub resource_limits: ComposeResourceLimits,
    /// The Docker entrypoint override, if any.
    pub entrypoint: Option<Vec<String>>,
    /// The Docker command override, if any.
    pub command: Option<Vec<String>>,
    /// Number of replicas (from `deploy.replicas` in v3, default 1).
    pub replicas: u32,
    /// Extra fields we don't explicitly model (preserved for round-tripping).
    pub extra: HashMap<String, Value>,
}

impl Default for ComposeService {
    fn default() -> Self {
        Self {
            name: String::new(),
            image: None,
            build_context: None,
            version: None,
            container_name: None,
            hostname: None,
            restart: RestartPolicy::default(),
            dependencies: Vec::new(),
            health_check: None,
            environment: HashMap::new(),
            labels: HashMap::new(),
            ports: Vec::new(),
            volumes: Vec::new(),
            networks: Vec::new(),
            resource_limits: ComposeResourceLimits::default(),
            entrypoint: None,
            command: None,
            replicas: 1,
            extra: HashMap::new(),
        }
    }
}

impl ComposeService {
    /// Returns the image tag, if the service has an image with a tag.
    pub fn image_tag(&self) -> Option<&str> {
        self.image.as_deref().and_then(|img| {
            let no_digest = img.split('@').next().unwrap_or(img);
            no_digest.rfind(':').map(|pos| {
                let candidate = &no_digest[pos + 1..];
                if candidate.contains('/') { "" } else { candidate }
            }).filter(|t| !t.is_empty())
        })
    }

    /// Returns the image repository (without registry or tag).
    pub fn image_repository(&self) -> Option<String> {
        self.image.as_deref().map(|img| {
            let no_digest = img.split('@').next().unwrap_or(img);
            let name_part = if let Some(pos) = no_digest.rfind(':') {
                let after = &no_digest[pos + 1..];
                if after.contains('/') { no_digest } else { &no_digest[..pos] }
            } else {
                no_digest
            };
            let parts: Vec<&str> = name_part.splitn(2, '/').collect();
            if parts.len() == 2
                && (parts[0].contains('.') || parts[0].contains(':') || parts[0] == "localhost")
            {
                parts[1].to_string()
            } else {
                name_part.to_string()
            }
        })
    }

    /// Returns the names of services this service depends on.
    pub fn dependency_names(&self) -> Vec<String> {
        self.dependencies.iter().map(|d| d.service.clone()).collect()
    }

    /// Returns the value of a specific label, if present.
    pub fn label(&self, key: &str) -> Option<&str> {
        self.labels.get(key).map(|v| v.as_str())
    }

    /// Returns environment variables whose names match a version-like pattern.
    pub fn version_env_vars(&self) -> HashMap<String, String> {
        self.environment
            .iter()
            .filter(|(k, _)| {
                let lower = k.to_lowercase();
                lower.contains("version") || lower.contains("_ver") || lower.ends_with("_tag")
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Checks whether this service declares a health check.
    pub fn has_health_check(&self) -> bool {
        self.health_check
            .as_ref()
            .map(|hc| !hc.disable && !hc.test.is_empty())
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Top-level network / volume definitions
// ---------------------------------------------------------------------------

/// A top-level named network defined in the Compose file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeNetworkDef {
    pub name: String,
    pub driver: Option<String>,
    pub external: bool,
    pub labels: HashMap<String, String>,
}

/// A top-level named volume defined in the Compose file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeVolumeDef {
    pub name: String,
    pub driver: Option<String>,
    pub external: bool,
    pub labels: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Compose file
// ---------------------------------------------------------------------------

/// A fully parsed Docker Compose file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeFile {
    /// Detected format version.
    pub format_version: ComposeFormatVersion,
    /// Raw version string from the file (e.g. `"3.8"`).
    pub raw_version: Option<String>,
    /// Service definitions keyed by service name.
    pub services: Vec<ComposeService>,
    /// Top-level network definitions.
    pub networks: Vec<ComposeNetworkDef>,
    /// Top-level volume definitions.
    pub volumes: Vec<ComposeVolumeDef>,
    /// The dependency graph: service name → list of dependency names.
    pub dependency_graph: HashMap<String, Vec<String>>,
}

impl ComposeFile {
    /// Returns a service by name, if it exists.
    pub fn service(&self, name: &str) -> Option<&ComposeService> {
        self.services.iter().find(|s| s.name == name)
    }

    /// Returns services that have no dependencies (the "roots" of the graph).
    pub fn root_services(&self) -> Vec<&ComposeService> {
        self.services
            .iter()
            .filter(|s| s.dependencies.is_empty())
            .collect()
    }

    /// Returns services in topological order (dependencies first).
    ///
    /// If the graph contains cycles, services involved in cycles are appended
    /// at the end in the order they were encountered.
    pub fn topological_order(&self) -> Vec<&ComposeService> {
        let mut visited: HashMap<&str, bool> = HashMap::new();
        let mut order: Vec<&str> = Vec::new();

        for svc in &self.services {
            if !visited.contains_key(svc.name.as_str()) {
                self.topo_visit(svc.name.as_str(), &mut visited, &mut order);
            }
        }

        order
            .iter()
            .filter_map(|name| self.service(name))
            .collect()
    }

    fn topo_visit<'a>(
        &'a self,
        name: &'a str,
        visited: &mut HashMap<&'a str, bool>,
        order: &mut Vec<&'a str>,
    ) {
        if let Some(&in_progress) = visited.get(name) {
            if in_progress {
                tracing::warn!(service = name, "cycle detected in Compose dependency graph");
            }
            return;
        }
        visited.insert(name, true); // mark in-progress
        if let Some(svc) = self.service(name) {
            for dep in &svc.dependencies {
                self.topo_visit(dep.service.as_str(), visited, order);
            }
        }
        visited.insert(name, false); // mark done
        order.push(name);
    }

    /// Returns the total number of services.
    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    /// Returns all unique image references across all services.
    pub fn all_images(&self) -> Vec<String> {
        let mut imgs: Vec<String> = self
            .services
            .iter()
            .filter_map(|s| s.image.clone())
            .collect();
        imgs.sort();
        imgs.dedup();
        imgs
    }
}

// ---------------------------------------------------------------------------
// Compose parser
// ---------------------------------------------------------------------------

/// Parses Docker Compose YAML files into structured [`ComposeFile`] objects.
///
/// Supports both v2 and v3 Compose formats, as well as the modern Compose
/// Specification (no version field).
pub struct ComposeParser;

impl ComposeParser {
    /// Parse a Docker Compose YAML string into a [`ComposeFile`].
    pub fn parse(yaml: &str) -> Result<ComposeFile> {
        let value: Value = serde_yaml::from_str(yaml).map_err(|e| SafeStepError::K8sError {
            message: format!("Compose YAML parse error: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })?;

        if !value.is_object() {
            return Err(SafeStepError::K8sError {
                message: "Compose file must be a YAML mapping".into(),
                resource: None,
                namespace: None,
                context: None,
            });
        }

        let (format_version, raw_version) = Self::detect_version(&value);
        tracing::debug!(?format_version, ?raw_version, "detected Compose format version");

        let services = Self::parse_services(&value, format_version)?;
        let networks = Self::parse_networks(&value);
        let volumes = Self::parse_volumes(&value);
        let dependency_graph = Self::build_dependency_graph(&services);

        Ok(ComposeFile {
            format_version,
            raw_version,
            services,
            networks,
            volumes,
            dependency_graph,
        })
    }

    /// Parse a Docker Compose YAML file from a byte slice.
    pub fn parse_bytes(bytes: &[u8]) -> Result<ComposeFile> {
        let yaml = std::str::from_utf8(bytes).map_err(|e| SafeStepError::K8sError {
            message: format!("Compose file is not valid UTF-8: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })?;
        Self::parse(yaml)
    }

    // -- version detection ---------------------------------------------------

    fn detect_version(root: &Value) -> (ComposeFormatVersion, Option<String>) {
        let raw = root
            .get("version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let version = match raw.as_deref() {
            Some(v) if v.starts_with('2') => ComposeFormatVersion::V2,
            Some(v) if v.starts_with('3') => ComposeFormatVersion::V3,
            Some(_) => ComposeFormatVersion::Modern,
            None => ComposeFormatVersion::Modern,
        };

        (version, raw)
    }

    // -- service parsing -----------------------------------------------------

    fn parse_services(root: &Value, fmt: ComposeFormatVersion) -> Result<Vec<ComposeService>> {
        let services_value = root.get("services").unwrap_or(root);
        let map = match services_value.as_object() {
            Some(m) => m,
            None => {
                return Err(SafeStepError::K8sError {
                    message: "Compose `services` must be a mapping".into(),
                    resource: None,
                    namespace: None,
                    context: None,
                });
            }
        };

        let mut services = Vec::new();
        for (name, def) in map {
            // Skip top-level keys that aren't service definitions.
            if is_top_level_key(name) {
                continue;
            }
            let svc = Self::parse_single_service(name, def, fmt)?;
            services.push(svc);
        }

        Ok(services)
    }

    fn parse_single_service(
        name: &str,
        def: &Value,
        fmt: ComposeFormatVersion,
    ) -> Result<ComposeService> {
        let mut svc = ComposeService {
            name: name.to_string(),
            ..Default::default()
        };

        // image
        svc.image = def.get("image").and_then(|v| v.as_str()).map(|s| s.to_string());

        // build context
        svc.build_context = match def.get("build") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(obj) => obj.get("context").and_then(|v| v.as_str()).map(|s| s.to_string()),
            None => None,
        };

        // version from image tag
        if let Some(img) = &svc.image {
            svc.version = VersionExtractor::from_image_tag(img);
        }

        // container_name, hostname
        svc.container_name = str_field(def, "container_name");
        svc.hostname = str_field(def, "hostname");

        // restart
        if let Some(r) = str_field(def, "restart") {
            svc.restart = RestartPolicy::parse(&r);
        }

        // depends_on
        svc.dependencies = Self::parse_depends_on(def)?;

        // healthcheck
        svc.health_check = Self::parse_healthcheck(def);

        // environment
        svc.environment = Self::parse_environment(def);

        // labels
        svc.labels = Self::parse_labels(def);

        // ports
        svc.ports = Self::parse_ports(def);

        // volumes
        svc.volumes = Self::parse_volumes_mounts(def);

        // networks
        svc.networks = Self::parse_service_networks(def);

        // entrypoint / command
        svc.entrypoint = Self::parse_string_or_list(def, "entrypoint");
        svc.command = Self::parse_string_or_list(def, "command");

        // resource limits + replicas from deploy (v3) or top-level (v2)
        svc.resource_limits = Self::parse_resource_limits(def, fmt);
        svc.replicas = Self::parse_replicas(def);

        // Collect remaining keys as extra
        if let Some(obj) = def.as_object() {
            let known_keys: &[&str] = &[
                "image", "build", "container_name", "hostname", "restart",
                "depends_on", "healthcheck", "environment", "labels", "ports",
                "volumes", "networks", "entrypoint", "command", "deploy",
                "mem_limit", "cpus", "cpu_shares", "memswap_limit",
            ];
            for (k, v) in obj {
                if !known_keys.contains(&k.as_str()) {
                    svc.extra.insert(k.clone(), v.clone());
                }
            }
        }

        Ok(svc)
    }

    // -- depends_on ----------------------------------------------------------

    /// Parse `depends_on` — supports both short list form and long map form.
    fn parse_depends_on(def: &Value) -> Result<Vec<ComposeDependency>> {
        let depends = match def.get("depends_on") {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        let mut deps = Vec::new();

        match depends {
            // Short form: depends_on: [db, redis]
            Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        deps.push(ComposeDependency {
                            service: s.to_string(),
                            condition: DependencyCondition::ServiceStarted,
                        });
                    }
                }
            }
            // Long form: depends_on: { db: { condition: service_healthy } }
            Value::Object(map) => {
                for (svc_name, cond_obj) in map {
                    let condition = cond_obj
                        .get("condition")
                        .and_then(|c| c.as_str())
                        .map(parse_dependency_condition)
                        .unwrap_or(DependencyCondition::ServiceStarted);
                    deps.push(ComposeDependency {
                        service: svc_name.clone(),
                        condition,
                    });
                }
            }
            _ => {}
        }

        Ok(deps)
    }

    // -- healthcheck ---------------------------------------------------------

    fn parse_healthcheck(def: &Value) -> Option<ComposeHealthCheck> {
        let hc = def.get("healthcheck")?;

        // Check for `disable: true`
        let disable = hc
            .get("disable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if disable {
            return Some(ComposeHealthCheck {
                disable: true,
                ..Default::default()
            });
        }

        let test = match hc.get("test") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            Some(Value::String(s)) => vec!["CMD-SHELL".to_string(), s.clone()],
            _ => Vec::new(),
        };

        Some(ComposeHealthCheck {
            test,
            interval: str_field(hc, "interval"),
            timeout: str_field(hc, "timeout"),
            retries: hc
                .get("retries")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            start_period: str_field(hc, "start_period"),
            disable: false,
        })
    }

    // -- environment ---------------------------------------------------------

    /// Parse `environment` — supports both map and list forms.
    fn parse_environment(def: &Value) -> HashMap<String, String> {
        let env = match def.get("environment") {
            Some(v) => v,
            None => return HashMap::new(),
        };
        parse_kv_field(env)
    }

    // -- labels --------------------------------------------------------------

    /// Parse `labels` — supports both map and list forms.
    fn parse_labels(def: &Value) -> HashMap<String, String> {
        let labels = match def.get("labels") {
            Some(v) => v,
            None => return HashMap::new(),
        };
        parse_kv_field(labels)
    }

    // -- ports ---------------------------------------------------------------

    fn parse_ports(def: &Value) -> Vec<ComposePort> {
        let ports = match def.get("ports").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return Vec::new(),
        };

        ports
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => parse_port_string(s),
                Value::Object(obj) => parse_port_object(obj),
                _ => None,
            })
            .collect()
    }

    // -- volumes -------------------------------------------------------------

    fn parse_volumes_mounts(def: &Value) -> Vec<ComposeVolume> {
        let vols = match def.get("volumes").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return Vec::new(),
        };

        vols.iter()
            .filter_map(|v| match v {
                Value::String(s) => parse_volume_string(s),
                Value::Object(obj) => parse_volume_object(obj),
                _ => None,
            })
            .collect()
    }

    // -- networks (service-level) --------------------------------------------

    fn parse_service_networks(def: &Value) -> Vec<ComposeNetworkRef> {
        let nets = match def.get("networks") {
            Some(v) => v,
            None => return Vec::new(),
        };

        match nets {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(|name| ComposeNetworkRef {
                    name: name.to_string(),
                    aliases: Vec::new(),
                })
                .collect(),
            Value::Object(map) => map
                .iter()
                .map(|(name, cfg)| {
                    let aliases = cfg
                        .get("aliases")
                        .and_then(|a| a.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    ComposeNetworkRef {
                        name: name.clone(),
                        aliases,
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    // -- string-or-list helper -----------------------------------------------

    fn parse_string_or_list(def: &Value, key: &str) -> Option<Vec<String>> {
        match def.get(key)? {
            Value::String(s) => Some(vec![s.clone()]),
            Value::Array(arr) => {
                let items: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                if items.is_empty() {
                    None
                } else {
                    Some(items)
                }
            }
            _ => None,
        }
    }

    // -- resource limits / replicas ------------------------------------------

    fn parse_resource_limits(def: &Value, fmt: ComposeFormatVersion) -> ComposeResourceLimits {
        match fmt {
            ComposeFormatVersion::V3 | ComposeFormatVersion::Modern => {
                Self::parse_deploy_resources(def)
            }
            ComposeFormatVersion::V2 => Self::parse_v2_resources(def),
        }
    }

    fn parse_deploy_resources(def: &Value) -> ComposeResourceLimits {
        let deploy = match def.get("deploy") {
            Some(d) => d,
            None => return ComposeResourceLimits::default(),
        };
        let resources = match deploy.get("resources") {
            Some(r) => r,
            None => return ComposeResourceLimits::default(),
        };

        ComposeResourceLimits {
            cpu_limit: resources
                .get("limits")
                .and_then(|l| l.get("cpus"))
                .and_then(|v| value_to_string(v)),
            memory_limit: resources
                .get("limits")
                .and_then(|l| l.get("memory"))
                .and_then(|v| value_to_string(v)),
            cpu_reservation: resources
                .get("reservations")
                .and_then(|r| r.get("cpus"))
                .and_then(|v| value_to_string(v)),
            memory_reservation: resources
                .get("reservations")
                .and_then(|r| r.get("memory"))
                .and_then(|v| value_to_string(v)),
        }
    }

    fn parse_v2_resources(def: &Value) -> ComposeResourceLimits {
        ComposeResourceLimits {
            cpu_limit: def.get("cpus").and_then(|v| value_to_string(v)),
            memory_limit: def
                .get("mem_limit")
                .and_then(|v| value_to_string(v)),
            cpu_reservation: def.get("cpu_shares").and_then(|v| value_to_string(v)),
            memory_reservation: def
                .get("memswap_limit")
                .and_then(|v| value_to_string(v)),
        }
    }

    fn parse_replicas(def: &Value) -> u32 {
        def.get("deploy")
            .and_then(|d| d.get("replicas"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(1)
    }

    // -- top-level networks / volumes ----------------------------------------

    fn parse_networks(root: &Value) -> Vec<ComposeNetworkDef> {
        let nets = match root.get("networks").and_then(|v| v.as_object()) {
            Some(m) => m,
            None => return Vec::new(),
        };

        nets.iter()
            .map(|(name, cfg)| ComposeNetworkDef {
                name: name.clone(),
                driver: cfg.get("driver").and_then(|v| v.as_str()).map(|s| s.to_string()),
                external: cfg.get("external").and_then(|v| v.as_bool()).unwrap_or(false),
                labels: cfg
                    .get("labels")
                    .map(parse_kv_field)
                    .unwrap_or_default(),
            })
            .collect()
    }

    fn parse_volumes(root: &Value) -> Vec<ComposeVolumeDef> {
        let vols = match root.get("volumes").and_then(|v| v.as_object()) {
            Some(m) => m,
            None => return Vec::new(),
        };

        vols.iter()
            .map(|(name, cfg)| {
                let cfg = if cfg.is_null() {
                    &Value::Object(serde_json::Map::new())
                } else {
                    cfg
                };
                ComposeVolumeDef {
                    name: name.clone(),
                    driver: cfg.get("driver").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    external: cfg.get("external").and_then(|v| v.as_bool()).unwrap_or(false),
                    labels: cfg
                        .get("labels")
                        .map(parse_kv_field)
                        .unwrap_or_default(),
                }
            })
            .collect()
    }

    // -- dependency graph ----------------------------------------------------

    fn build_dependency_graph(services: &[ComposeService]) -> HashMap<String, Vec<String>> {
        services
            .iter()
            .map(|s| (s.name.clone(), s.dependency_names()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Compose version extractor
// ---------------------------------------------------------------------------

/// Converts parsed Docker Compose data into SafeStep's service descriptors
/// and version-product graph for deployment planning.
pub struct ComposeVersionExtractor;

impl ComposeVersionExtractor {
    /// Convert a [`ComposeFile`] into a list of [`ServiceDescriptor`]s.
    pub fn extract_service_descriptors(
        compose: &ComposeFile,
    ) -> Vec<ServiceDescriptor> {
        compose
            .services
            .iter()
            .map(|svc| Self::service_to_descriptor(svc))
            .collect()
    }

    /// Convert a single [`ComposeService`] into a [`ServiceDescriptor`].
    pub fn service_to_descriptor(svc: &ComposeService) -> ServiceDescriptor {
        let containers = Self::build_container_infos(svc);
        let ports: Vec<u16> = svc.ports.iter().map(|p| p.container).collect();
        let version = svc.version.clone().or_else(|| {
            Self::version_from_env(svc).or_else(|| Self::version_from_labels(svc))
        });

        ServiceDescriptor {
            name: svc.name.clone(),
            namespace: "compose".to_string(),
            kind: "ComposeService".to_string(),
            version,
            replicas: svc.replicas,
            containers,
            labels: svc.labels.clone(),
            total_cpu_request: 0.0,
            total_memory_request: 0.0,
            total_cpu_limit: 0.0,
            total_memory_limit: 0.0,
            ports,
            dependencies: svc.dependency_names(),
        }
    }

    /// Build [`ContainerInfo`] entries for a Compose service.
    fn build_container_infos(svc: &ComposeService) -> Vec<ContainerInfo> {
        let image_str = svc.image.clone().unwrap_or_default();
        if image_str.is_empty() {
            return Vec::new();
        }

        let (registry, repository, tag) = parse_image_ref_compose(&image_str);

        vec![ContainerInfo {
            name: svc.name.clone(),
            image: image_str,
            registry,
            repository,
            tag,
            cpu_request: 0.0,
            cpu_limit: 0.0,
            memory_request: 0.0,
            memory_limit: 0.0,
        }]
    }

    /// Attempt to extract a version from well-known environment variables.
    fn version_from_env(svc: &ComposeService) -> Option<ServiceVersion> {
        let version_keys = [
            "VERSION",
            "APP_VERSION",
            "IMAGE_TAG",
            "SERVICE_VERSION",
            "RELEASE_VERSION",
        ];
        for key in &version_keys {
            if let Some(val) = svc.environment.get(*key) {
                if let Some(v) = VersionExtractor::from_tag(val) {
                    return Some(v);
                }
            }
        }
        // Also check pattern-matched keys
        for (key, val) in &svc.environment {
            let lower = key.to_lowercase();
            if (lower.contains("version") || lower.ends_with("_tag") || lower.ends_with("_ver"))
                && !val.is_empty()
            {
                if let Some(v) = VersionExtractor::from_tag(val) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Attempt to extract a version from well-known labels.
    fn version_from_labels(svc: &ComposeService) -> Option<ServiceVersion> {
        let label_keys = [
            "org.opencontainers.image.version",
            "com.safestep.version",
            "app.version",
            "version",
        ];
        for key in &label_keys {
            if let Some(val) = svc.labels.get(*key) {
                if let Some(v) = VersionExtractor::from_tag(val) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Extract a complete dependency graph from a [`ComposeFile`] as a map
    /// of service name → list of `(dependency_name, condition)`.
    pub fn extract_dependency_graph(
        compose: &ComposeFile,
    ) -> HashMap<String, Vec<(String, DependencyCondition)>> {
        compose
            .services
            .iter()
            .map(|svc| {
                let deps = svc
                    .dependencies
                    .iter()
                    .map(|d| (d.service.clone(), d.condition))
                    .collect();
                (svc.name.clone(), deps)
            })
            .collect()
    }

    /// Extract compatibility metadata from labels.
    ///
    /// Looks for labels with the prefix `com.safestep.compat.` and returns
    /// a map of constraint-name → constraint-value per service.
    pub fn extract_compatibility_metadata(
        compose: &ComposeFile,
    ) -> HashMap<String, HashMap<String, String>> {
        const PREFIX: &str = "com.safestep.compat.";

        compose
            .services
            .iter()
            .filter_map(|svc| {
                let compat: HashMap<String, String> = svc
                    .labels
                    .iter()
                    .filter(|(k, _)| k.starts_with(PREFIX))
                    .map(|(k, v)| (k[PREFIX.len()..].to_string(), v.clone()))
                    .collect();
                if compat.is_empty() {
                    None
                } else {
                    Some((svc.name.clone(), compat))
                }
            })
            .collect()
    }

    /// Produce a mapping from service name → set of version strings discovered
    /// from images, env vars, and labels.
    pub fn collect_all_versions(
        compose: &ComposeFile,
    ) -> HashMap<String, Vec<ServiceVersion>> {
        compose
            .services
            .iter()
            .map(|svc| {
                let mut versions = Vec::new();

                // From image tag
                if let Some(v) = &svc.version {
                    versions.push(v.clone());
                }

                // From env vars
                for (_, val) in svc.version_env_vars() {
                    if let Some(v) = VersionExtractor::from_tag(&val) {
                        if !versions.iter().any(|existing| existing.raw == v.raw) {
                            versions.push(v);
                        }
                    }
                }

                // From labels
                if let Some(v) = Self::version_from_labels(svc) {
                    if !versions.iter().any(|existing| existing.raw == v.raw) {
                        versions.push(v);
                    }
                }

                (svc.name.clone(), versions)
            })
            .collect()
    }
}

// ===========================================================================
// Private helper functions
// ===========================================================================

/// Returns `true` if the key is a well-known Compose top-level key
/// (and therefore not a service definition).
fn is_top_level_key(key: &str) -> bool {
    matches!(
        key,
        "version" | "services" | "networks" | "volumes" | "configs" | "secrets" | "x-"
    ) || key.starts_with("x-")
}

/// Extract a string field from a YAML value.
fn str_field(val: &Value, key: &str) -> Option<String> {
    val.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Convert a `Value` to a `String`, handling both string and number forms.
fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Parse a key-value field that can be either a YAML mapping or a list of
/// `KEY=VALUE` strings.
fn parse_kv_field(val: &Value) -> HashMap<String, String> {
    let mut map = HashMap::new();
    match val {
        Value::Object(obj) => {
            for (k, v) in obj {
                let val_str = match v {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                map.insert(k.clone(), val_str);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    if let Some((k, v)) = s.split_once('=') {
                        map.insert(k.to_string(), v.to_string());
                    } else {
                        // Variable reference without a value
                        map.insert(s.to_string(), String::new());
                    }
                }
            }
        }
        _ => {}
    }
    map
}

/// Parse a dependency condition string into a [`DependencyCondition`].
fn parse_dependency_condition(s: &str) -> DependencyCondition {
    match s {
        "service_healthy" => DependencyCondition::ServiceHealthy,
        "service_completed_successfully" => DependencyCondition::ServiceCompletedSuccessfully,
        _ => DependencyCondition::ServiceStarted,
    }
}

/// Parse a port string like `"8080:80"`, `"8080:80/udp"`, or `"80"`.
fn parse_port_string(s: &str) -> Option<ComposePort> {
    // Strip protocol suffix
    let (port_part, protocol) = if let Some(idx) = s.rfind('/') {
        (&s[..idx], s[idx + 1..].to_string())
    } else {
        (s, "tcp".to_string())
    };

    if let Some((host_str, container_str)) = port_part.rsplit_once(':') {
        // Handle IP binding: "127.0.0.1:8080:80"
        let host_port_str = if host_str.contains(':') {
            host_str.rsplit_once(':').map(|(_, p)| p).unwrap_or(host_str)
        } else {
            host_str
        };
        let host = host_port_str.parse::<u16>().ok();
        let container = container_str.parse::<u16>().ok()?;
        Some(ComposePort {
            host,
            container,
            protocol,
        })
    } else {
        let container = port_part.parse::<u16>().ok()?;
        Some(ComposePort {
            host: None,
            container,
            protocol,
        })
    }
}

/// Parse a port object (long-form).
fn parse_port_object(obj: &serde_json::Map<String, Value>) -> Option<ComposePort> {
    let target = obj.get("target").and_then(|v| v.as_u64())? as u16;
    let published = obj.get("published").and_then(|v| v.as_u64()).map(|v| v as u16);
    let protocol = obj
        .get("protocol")
        .and_then(|v| v.as_str())
        .unwrap_or("tcp")
        .to_string();
    Some(ComposePort {
        host: published,
        container: target,
        protocol,
    })
}

/// Parse a short-form volume string like `"./data:/var/lib/data:ro"`.
fn parse_volume_string(s: &str) -> Option<ComposeVolume> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    match parts.len() {
        1 => Some(ComposeVolume {
            source: String::new(),
            target: parts[0].to_string(),
            read_only: false,
        }),
        2 => Some(ComposeVolume {
            source: parts[0].to_string(),
            target: parts[1].to_string(),
            read_only: false,
        }),
        3 => Some(ComposeVolume {
            source: parts[0].to_string(),
            target: parts[1].to_string(),
            read_only: parts[2] == "ro",
        }),
        _ => None,
    }
}

/// Parse a long-form volume object.
fn parse_volume_object(obj: &serde_json::Map<String, Value>) -> Option<ComposeVolume> {
    let target = obj.get("target").and_then(|v| v.as_str())?.to_string();
    let source = obj
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let read_only = obj
        .get("read_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(ComposeVolume {
        source,
        target,
        read_only,
    })
}

/// Parse a Docker image reference into (registry, repository, tag).
fn parse_image_ref_compose(image: &str) -> (Option<String>, String, Option<String>) {
    let image_no_digest = image.split('@').next().unwrap_or(image);

    let (name_part, tag) = if let Some(colon_pos) = image_no_digest.rfind(':') {
        let after_colon = &image_no_digest[colon_pos + 1..];
        if after_colon.contains('/') {
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

    let parts: Vec<&str> = name_part.splitn(2, '/').collect();
    let (registry, repository) = if parts.len() == 2
        && (parts[0].contains('.') || parts[0].contains(':') || parts[0] == "localhost")
    {
        (Some(parts[0].to_string()), parts[1].to_string())
    } else {
        (None, name_part.to_string())
    };

    (registry, repository, tag)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- format detection ----------------------------------------------------

    #[test]
    fn test_detect_version_v2() {
        let yaml = r#"
version: "2.4"
services:
  web:
    image: nginx:1.25
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        assert_eq!(compose.format_version, ComposeFormatVersion::V2);
        assert_eq!(compose.raw_version.as_deref(), Some("2.4"));
    }

    #[test]
    fn test_detect_version_v3() {
        let yaml = r#"
version: "3.8"
services:
  web:
    image: nginx:1.25
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        assert_eq!(compose.format_version, ComposeFormatVersion::V3);
        assert_eq!(compose.raw_version.as_deref(), Some("3.8"));
    }

    #[test]
    fn test_detect_version_modern() {
        let yaml = r#"
services:
  web:
    image: nginx:1.25
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        assert_eq!(compose.format_version, ComposeFormatVersion::Modern);
        assert!(compose.raw_version.is_none());
    }

    // -- service parsing -----------------------------------------------------

    #[test]
    fn test_parse_basic_service() {
        let yaml = r#"
version: "3"
services:
  api:
    image: myapp/api:2.1.0
    container_name: api-server
    hostname: api
    restart: always
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        assert_eq!(compose.services.len(), 1);

        let svc = &compose.services[0];
        assert_eq!(svc.name, "api");
        assert_eq!(svc.image.as_deref(), Some("myapp/api:2.1.0"));
        assert_eq!(svc.container_name.as_deref(), Some("api-server"));
        assert_eq!(svc.hostname.as_deref(), Some("api"));
        assert_eq!(svc.restart, RestartPolicy::Always);

        let v = svc.version.as_ref().unwrap();
        assert_eq!(v.major, Some(2));
        assert_eq!(v.minor, Some(1));
        assert_eq!(v.patch, Some(0));
    }

    // -- depends_on ----------------------------------------------------------

    #[test]
    fn test_depends_on_short_form() {
        let yaml = r#"
services:
  web:
    image: nginx
    depends_on:
      - db
      - redis
  db:
    image: postgres:16
  redis:
    image: redis:7
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert_eq!(web.dependencies.len(), 2);
        assert_eq!(web.dependencies[0].service, "db");
        assert_eq!(web.dependencies[0].condition, DependencyCondition::ServiceStarted);
    }

    #[test]
    fn test_depends_on_long_form_with_conditions() {
        let yaml = r#"
services:
  web:
    image: nginx
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
      migrate:
        condition: service_completed_successfully
  db:
    image: postgres:16
  redis:
    image: redis:7
  migrate:
    image: myapp/migrate:1.0
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();

        assert_eq!(web.dependencies.len(), 3);

        let db_dep = web.dependencies.iter().find(|d| d.service == "db").unwrap();
        assert_eq!(db_dep.condition, DependencyCondition::ServiceHealthy);

        let migrate_dep = web
            .dependencies
            .iter()
            .find(|d| d.service == "migrate")
            .unwrap();
        assert_eq!(
            migrate_dep.condition,
            DependencyCondition::ServiceCompletedSuccessfully
        );
    }

    // -- healthcheck ---------------------------------------------------------

    #[test]
    fn test_healthcheck_parsed() {
        let yaml = r#"
services:
  db:
    image: postgres:16
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let db = compose.service("db").unwrap();
        assert!(db.has_health_check());

        let hc = db.health_check.as_ref().unwrap();
        assert_eq!(hc.test, vec!["CMD-SHELL", "pg_isready -U postgres"]);
        assert_eq!(hc.interval.as_deref(), Some("10s"));
        assert_eq!(hc.timeout.as_deref(), Some("5s"));
        assert_eq!(hc.retries, Some(5));
        assert_eq!(hc.start_period.as_deref(), Some("30s"));
    }

    #[test]
    fn test_healthcheck_disabled() {
        let yaml = r#"
services:
  web:
    image: nginx
    healthcheck:
      disable: true
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert!(!web.has_health_check());
        assert!(web.health_check.as_ref().unwrap().disable);
    }

    // -- environment ---------------------------------------------------------

    #[test]
    fn test_environment_map_form() {
        let yaml = r#"
services:
  app:
    image: myapp:1.0
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      VERSION: 2.3.1
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(app.environment.get("DB_HOST").unwrap(), "postgres");
        assert_eq!(app.environment.get("DB_PORT").unwrap(), "5432");
        assert_eq!(app.environment.get("VERSION").unwrap(), "2.3.1");
    }

    #[test]
    fn test_environment_list_form() {
        let yaml = r#"
services:
  app:
    image: myapp:1.0
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(app.environment.get("DB_HOST").unwrap(), "postgres");
        assert_eq!(app.environment.get("DB_PORT").unwrap(), "5432");
    }

    // -- labels --------------------------------------------------------------

    #[test]
    fn test_labels_map_form() {
        let yaml = r#"
services:
  app:
    image: myapp:1.0
    labels:
      com.safestep.compat.min-redis: "6.0"
      org.opencontainers.image.version: "1.0.0"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(
            app.labels.get("com.safestep.compat.min-redis").unwrap(),
            "6.0"
        );
    }

    #[test]
    fn test_labels_list_form() {
        let yaml = r#"
services:
  app:
    image: myapp:1.0
    labels:
      - "com.safestep.version=1.0.0"
      - "tier=frontend"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(
            app.labels.get("com.safestep.version").unwrap(),
            "1.0.0"
        );
        assert_eq!(app.labels.get("tier").unwrap(), "frontend");
    }

    // -- ports ---------------------------------------------------------------

    #[test]
    fn test_ports_short_form() {
        let yaml = r#"
services:
  web:
    image: nginx
    ports:
      - "8080:80"
      - "443:443/tcp"
      - "9090"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert_eq!(web.ports.len(), 3);
        assert_eq!(web.ports[0].host, Some(8080));
        assert_eq!(web.ports[0].container, 80);
        assert_eq!(web.ports[1].container, 443);
        assert_eq!(web.ports[2].host, None);
        assert_eq!(web.ports[2].container, 9090);
    }

    #[test]
    fn test_ports_long_form() {
        let yaml = r#"
services:
  web:
    image: nginx
    ports:
      - target: 80
        published: 8080
        protocol: tcp
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert_eq!(web.ports.len(), 1);
        assert_eq!(web.ports[0].host, Some(8080));
        assert_eq!(web.ports[0].container, 80);
        assert_eq!(web.ports[0].protocol, "tcp");
    }

    // -- volumes -------------------------------------------------------------

    #[test]
    fn test_volumes_short_form() {
        let yaml = r#"
services:
  db:
    image: postgres:16
    volumes:
      - "pgdata:/var/lib/postgresql/data"
      - "./config:/etc/config:ro"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let db = compose.service("db").unwrap();
        assert_eq!(db.volumes.len(), 2);
        assert_eq!(db.volumes[0].source, "pgdata");
        assert_eq!(db.volumes[0].target, "/var/lib/postgresql/data");
        assert!(!db.volumes[0].read_only);
        assert_eq!(db.volumes[1].source, "./config");
        assert!(db.volumes[1].read_only);
    }

    // -- replicas / deploy ---------------------------------------------------

    #[test]
    fn test_v3_deploy_replicas_and_resources() {
        let yaml = r#"
version: "3.8"
services:
  web:
    image: nginx:1.25
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
        reservations:
          cpus: "0.25"
          memory: 128M
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert_eq!(web.replicas, 3);
        assert_eq!(web.resource_limits.cpu_limit.as_deref(), Some("0.5"));
        assert_eq!(web.resource_limits.memory_limit.as_deref(), Some("256M"));
        assert_eq!(
            web.resource_limits.cpu_reservation.as_deref(),
            Some("0.25")
        );
    }

    // -- networks / top-level ------------------------------------------------

    #[test]
    fn test_top_level_networks() {
        let yaml = r#"
services:
  web:
    image: nginx
    networks:
      - frontend
      - backend
networks:
  frontend:
    driver: bridge
  backend:
    external: true
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        assert_eq!(compose.networks.len(), 2);

        let frontend = compose.networks.iter().find(|n| n.name == "frontend").unwrap();
        assert_eq!(frontend.driver.as_deref(), Some("bridge"));
        assert!(!frontend.external);

        let backend = compose.networks.iter().find(|n| n.name == "backend").unwrap();
        assert!(backend.external);
    }

    #[test]
    fn test_service_network_aliases() {
        let yaml = r#"
services:
  web:
    image: nginx
    networks:
      frontend:
        aliases:
          - web-alias
          - web-alt
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let web = compose.service("web").unwrap();
        assert_eq!(web.networks.len(), 1);
        assert_eq!(web.networks[0].name, "frontend");
        assert_eq!(web.networks[0].aliases, vec!["web-alias", "web-alt"]);
    }

    // -- build context -------------------------------------------------------

    #[test]
    fn test_build_context_string() {
        let yaml = r#"
services:
  app:
    build: ./app
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(app.build_context.as_deref(), Some("./app"));
        assert!(app.image.is_none());
    }

    #[test]
    fn test_build_context_object() {
        let yaml = r#"
services:
  app:
    build:
      context: ./app
      dockerfile: Dockerfile.prod
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(app.build_context.as_deref(), Some("./app"));
    }

    // -- ComposeVersionExtractor --------------------------------------------

    #[test]
    fn test_service_descriptor_conversion() {
        let yaml = r#"
version: "3"
services:
  api:
    image: ghcr.io/myorg/api:3.2.1
    depends_on:
      - db
    ports:
      - "8080:80"
    labels:
      com.safestep.compat.min-db: "16"
  db:
    image: postgres:16.1
    environment:
      POSTGRES_VERSION: "16.1"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let descriptors = ComposeVersionExtractor::extract_service_descriptors(&compose);

        assert_eq!(descriptors.len(), 2);

        let api = descriptors.iter().find(|d| d.name == "api").unwrap();
        assert_eq!(api.namespace, "compose");
        assert_eq!(api.kind, "ComposeService");
        assert_eq!(api.version.as_ref().unwrap().major, Some(3));
        assert_eq!(api.dependencies, vec!["db"]);
        assert_eq!(api.ports, vec![80]);

        let db = descriptors.iter().find(|d| d.name == "db").unwrap();
        assert_eq!(db.version.as_ref().unwrap().major, Some(16));
    }

    #[test]
    fn test_compatibility_metadata_extraction() {
        let yaml = r#"
services:
  api:
    image: myapp:1.0
    labels:
      com.safestep.compat.min-redis: "6.0"
      com.safestep.compat.max-postgres: "17"
      unrelated: "ignored"
  db:
    image: postgres:16
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let compat = ComposeVersionExtractor::extract_compatibility_metadata(&compose);

        assert!(compat.contains_key("api"));
        assert!(!compat.contains_key("db"));

        let api_compat = &compat["api"];
        assert_eq!(api_compat.get("min-redis").unwrap(), "6.0");
        assert_eq!(api_compat.get("max-postgres").unwrap(), "17");
        assert!(!api_compat.contains_key("unrelated"));
    }

    #[test]
    fn test_collect_all_versions() {
        let yaml = r#"
services:
  app:
    image: myapp:2.0.0
    environment:
      APP_VERSION: "2.0.0"
      CLIENT_TAG: "1.5.0"
    labels:
      org.opencontainers.image.version: "2.0.0"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let versions = ComposeVersionExtractor::collect_all_versions(&compose);
        let app_versions = &versions["app"];
        // image tag 2.0.0, CLIENT_TAG 1.5.0 (APP_VERSION duplicate of image)
        assert!(app_versions.len() >= 2);
        assert!(app_versions.iter().any(|v| v.raw == "2.0.0"));
        assert!(app_versions.iter().any(|v| v.raw == "1.5.0"));
    }

    #[test]
    fn test_version_from_env_fallback() {
        let yaml = r#"
services:
  app:
    image: myapp
    environment:
      SERVICE_VERSION: "4.1.2"
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let descriptors = ComposeVersionExtractor::extract_service_descriptors(&compose);
        let app = descriptors.iter().find(|d| d.name == "app").unwrap();
        let v = app.version.as_ref().unwrap();
        assert_eq!(v.major, Some(4));
        assert_eq!(v.minor, Some(1));
        assert_eq!(v.patch, Some(2));
    }

    // -- topological order ---------------------------------------------------

    #[test]
    fn test_topological_order() {
        let yaml = r#"
services:
  web:
    image: nginx
    depends_on:
      - api
  api:
    image: myapp:1.0
    depends_on:
      - db
      - redis
  db:
    image: postgres:16
  redis:
    image: redis:7
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let order: Vec<&str> = compose
            .topological_order()
            .iter()
            .map(|s| s.name.as_str())
            .collect();

        let db_pos = order.iter().position(|&s| s == "db").unwrap();
        let redis_pos = order.iter().position(|&s| s == "redis").unwrap();
        let api_pos = order.iter().position(|&s| s == "api").unwrap();
        let web_pos = order.iter().position(|&s| s == "web").unwrap();

        assert!(db_pos < api_pos);
        assert!(redis_pos < api_pos);
        assert!(api_pos < web_pos);
    }

    #[test]
    fn test_root_services() {
        let yaml = r#"
services:
  web:
    image: nginx
    depends_on:
      - api
  api:
    image: myapp:1.0
  db:
    image: postgres:16
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let roots: Vec<&str> = compose
            .root_services()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert!(roots.contains(&"api"));
        assert!(roots.contains(&"db"));
        assert!(!roots.contains(&"web"));
    }

    // -- dependency graph extraction -----------------------------------------

    #[test]
    fn test_dependency_graph_with_conditions() {
        let yaml = r#"
services:
  web:
    image: nginx
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
  db:
    image: postgres:16
  cache:
    image: redis:7
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let graph = ComposeVersionExtractor::extract_dependency_graph(&compose);
        let web_deps = &graph["web"];
        assert_eq!(web_deps.len(), 2);

        let db_dep = web_deps.iter().find(|(n, _)| n == "db").unwrap();
        assert_eq!(db_dep.1, DependencyCondition::ServiceHealthy);
    }

    // -- image helpers -------------------------------------------------------

    #[test]
    fn test_image_tag_extraction() {
        let svc = ComposeService {
            name: "test".into(),
            image: Some("ghcr.io/org/repo:v1.2.3".into()),
            ..Default::default()
        };
        assert_eq!(svc.image_tag(), Some("v1.2.3"));
    }

    #[test]
    fn test_image_repository_extraction() {
        let svc = ComposeService {
            name: "test".into(),
            image: Some("ghcr.io/org/repo:v1.2.3".into()),
            ..Default::default()
        };
        assert_eq!(svc.image_repository().as_deref(), Some("org/repo"));
    }

    #[test]
    fn test_all_images() {
        let yaml = r#"
services:
  web:
    image: nginx:1.25
  db:
    image: postgres:16
  app:
    build: ./app
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let imgs = compose.all_images();
        assert_eq!(imgs.len(), 2);
        assert!(imgs.contains(&"nginx:1.25".to_string()));
        assert!(imgs.contains(&"postgres:16".to_string()));
    }

    // -- entrypoint / command ------------------------------------------------

    #[test]
    fn test_entrypoint_and_command() {
        let yaml = r#"
services:
  app:
    image: myapp:1.0
    entrypoint: /entrypoint.sh
    command: ["serve", "--port", "8080"]
"#;
        let compose = ComposeParser::parse(yaml).unwrap();
        let app = compose.service("app").unwrap();
        assert_eq!(app.entrypoint, Some(vec!["/entrypoint.sh".to_string()]));
        assert_eq!(
            app.command,
            Some(vec!["serve".to_string(), "--port".to_string(), "8080".to_string()])
        );
    }

    // -- error handling ------------------------------------------------------

    #[test]
    fn test_invalid_yaml_returns_error() {
        let result = ComposeParser::parse("{{invalid yaml}}");
        assert!(result.is_err());
    }

    #[test]
    fn test_non_mapping_returns_error() {
        let result = ComposeParser::parse("- item1\n- item2\n");
        assert!(result.is_err());
    }

    // -- version env var helpers ---------------------------------------------

    #[test]
    fn test_version_env_vars() {
        let svc = ComposeService {
            name: "test".into(),
            environment: [
                ("DB_HOST".into(), "localhost".into()),
                ("APP_VERSION".into(), "1.0.0".into()),
                ("IMAGE_TAG".into(), "latest".into()),
                ("SOME_VER".into(), "2.0".into()),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let ver_vars = svc.version_env_vars();
        assert!(ver_vars.contains_key("APP_VERSION"));
        assert!(ver_vars.contains_key("IMAGE_TAG"));
        assert!(ver_vars.contains_key("SOME_VER"));
        assert!(!ver_vars.contains_key("DB_HOST"));
    }
}
