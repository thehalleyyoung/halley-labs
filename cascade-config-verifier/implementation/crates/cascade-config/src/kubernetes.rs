//! Kubernetes resource parsing for the CascadeVerify project.
//!
//! Provides parsers for core Kubernetes resource types: Deployments, Services,
//! Ingress, and ConfigMaps.  Multi-document YAML manifests are split and each
//! document is dispatched to the appropriate parser based on its `kind` field.
//!
//! Service dependency extraction, retry-policy annotation parsing, and label
//! selector matching are also provided for cross-resource analysis.

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{ObjectMeta, RetryPolicy, ServiceId};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Parsed Kubernetes Deployment resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Deployment {
    pub metadata: ObjectMeta,
    pub spec: DeploymentSpec,
}

/// The specification section of a Deployment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeploymentSpec {
    pub replicas: u32,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub strategy: DeploymentStrategy,
}

/// Rolling-update / recreate strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeploymentStrategy {
    pub strategy_type: String,
    pub max_unavailable: Option<String>,
    pub max_surge: Option<String>,
}

impl Default for DeploymentStrategy {
    fn default() -> Self {
        Self {
            strategy_type: "RollingUpdate".to_string(),
            max_unavailable: Some("25%".to_string()),
            max_surge: Some("25%".to_string()),
        }
    }
}

/// Label selector used by Deployments and Services.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LabelSelector {
    pub match_labels: IndexMap<String, String>,
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// A single expression clause inside a [`LabelSelector`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelSelectorRequirement {
    pub key: String,
    pub operator: String,
    pub values: Vec<String>,
}

/// Pod template embedded inside a Deployment spec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PodTemplateSpec {
    pub metadata: ObjectMeta,
    pub containers: Vec<ContainerSpec>,
    pub init_containers: Vec<ContainerSpec>,
    pub volumes: Vec<Volume>,
    pub service_account: Option<String>,
}

/// Specification of a single container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerSpec {
    pub name: String,
    pub image: String,
    pub ports: Vec<ContainerPort>,
    pub resources: ResourceRequirements,
    pub env: Vec<EnvVar>,
    pub probes: ContainerProbes,
    pub command: Vec<String>,
    pub args: Vec<String>,
    pub volume_mounts: Vec<VolumeMount>,
}

/// A named port exposed by a container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerPort {
    pub name: Option<String>,
    pub container_port: u16,
    pub protocol: String,
}

/// CPU / memory resource requests and limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResourceRequirements {
    pub cpu_request: Option<String>,
    pub cpu_limit: Option<String>,
    pub memory_request: Option<String>,
    pub memory_limit: Option<String>,
}

/// An environment variable definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvVar {
    pub name: String,
    pub value: Option<String>,
    pub value_from: Option<EnvVarSource>,
}

/// Reference-based env-var value source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvVarSource {
    pub config_map_key_ref: Option<KeyRef>,
    pub secret_key_ref: Option<KeyRef>,
    pub field_ref: Option<FieldRef>,
}

/// Reference to a key inside a ConfigMap or Secret.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeyRef {
    pub name: String,
    pub key: String,
}

/// Reference to a pod-level field (e.g. `status.podIP`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldRef {
    pub field_path: String,
}

/// Liveness / readiness / startup probes for a container.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ContainerProbes {
    pub liveness: Option<Probe>,
    pub readiness: Option<Probe>,
    pub startup: Option<Probe>,
}

/// A single health-check probe definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Probe {
    pub http_get: Option<HttpGetAction>,
    pub tcp_socket: Option<TcpSocketAction>,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
}

/// HTTP GET probe action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpGetAction {
    pub path: String,
    pub port: u16,
}

/// TCP socket probe action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpSocketAction {
    pub port: u16,
}

/// A volume that can be mounted into a container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Volume {
    pub name: String,
    pub volume_type: VolumeType,
}

/// The backing type for a [`Volume`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VolumeType {
    ConfigMap(String),
    Secret(String),
    EmptyDir,
    PersistentVolumeClaim(String),
}

/// Volume mount inside a container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VolumeMount {
    pub name: String,
    pub mount_path: String,
    pub read_only: bool,
}

/// Parsed Kubernetes Service resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KubeService {
    pub metadata: ObjectMeta,
    pub spec: ServiceSpec,
}

/// The specification section of a Service.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceSpec {
    pub service_type: String,
    pub selector: IndexMap<String, String>,
    pub ports: Vec<ServicePort>,
    pub cluster_ip: Option<String>,
}

/// A port definition on a Kubernetes Service.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServicePort {
    pub name: Option<String>,
    pub port: u16,
    pub target_port: u16,
    pub protocol: String,
    pub node_port: Option<u16>,
}

/// Parsed Kubernetes Ingress resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ingress {
    pub metadata: ObjectMeta,
    pub rules: Vec<IngressRule>,
    pub tls: Vec<IngressTLS>,
}

/// A single Ingress routing rule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngressRule {
    pub host: Option<String>,
    pub paths: Vec<IngressPath>,
}

/// A path entry inside an [`IngressRule`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngressPath {
    pub path: String,
    pub path_type: String,
    pub backend: IngressBackend,
}

/// Backend reference for an Ingress path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngressBackend {
    pub service_name: String,
    pub service_port: u16,
}

/// TLS termination configuration for an Ingress.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngressTLS {
    pub hosts: Vec<String>,
    pub secret_name: String,
}

/// Parsed Kubernetes ConfigMap resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigMap {
    pub metadata: ObjectMeta,
    pub data: IndexMap<String, String>,
    pub binary_data: IndexMap<String, Vec<u8>>,
}

/// A wrapper enum for heterogeneous Kubernetes resources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KubernetesResource {
    Deployment(Deployment),
    Service(KubeService),
    Ingress(Ingress),
    ConfigMap(ConfigMap),
    Unknown(String, serde_yaml::Value),
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Entry-point for parsing Kubernetes YAML manifests.
#[derive(Debug, Default)]
pub struct KubernetesParser;

impl KubernetesParser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        Self
    }

    // -- public API ----------------------------------------------------------

    /// Parse a YAML string into a [`Deployment`].
    pub fn parse_deployment(&self, yaml: &str) -> Result<Deployment> {
        let val: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("invalid YAML for Deployment")?;
        let metadata = parse_metadata(&val)?;
        let spec_val = &val["spec"];
        if spec_val.is_null() {
            bail!("Deployment is missing spec");
        }

        let replicas = spec_val["replicas"].as_u64().unwrap_or(1) as u32;

        let selector = parse_label_selector(&spec_val["selector"]);

        let template_val = &spec_val["template"];
        let template = parse_pod_template(template_val)?;

        let strategy = parse_strategy(&spec_val["strategy"]);

        Ok(Deployment {
            metadata,
            spec: DeploymentSpec {
                replicas,
                selector,
                template,
                strategy,
            },
        })
    }

    /// Parse a YAML string into a [`KubeService`].
    pub fn parse_service(&self, yaml: &str) -> Result<KubeService> {
        let val: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("invalid YAML for Service")?;
        let metadata = parse_metadata(&val)?;
        let spec_val = &val["spec"];
        if spec_val.is_null() {
            bail!("Service is missing spec");
        }

        let service_type = spec_val["type"]
            .as_str()
            .unwrap_or("ClusterIP")
            .to_string();

        let selector = parse_string_map(&spec_val["selector"]);

        let ports = parse_service_ports(&spec_val["ports"]);

        let cluster_ip = spec_val["clusterIP"].as_str().map(String::from);

        Ok(KubeService {
            metadata,
            spec: ServiceSpec {
                service_type,
                selector,
                ports,
                cluster_ip,
            },
        })
    }

    /// Parse a YAML string into an [`Ingress`].
    pub fn parse_ingress(&self, yaml: &str) -> Result<Ingress> {
        let val: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("invalid YAML for Ingress")?;
        let metadata = parse_metadata(&val)?;
        let spec_val = &val["spec"];
        if spec_val.is_null() {
            bail!("Ingress is missing spec");
        }

        let rules = parse_ingress_rules(&spec_val["rules"]);
        let tls = parse_ingress_tls(&spec_val["tls"]);

        Ok(Ingress {
            metadata,
            rules,
            tls,
        })
    }

    /// Parse a YAML string into a [`ConfigMap`].
    pub fn parse_config_map(&self, yaml: &str) -> Result<ConfigMap> {
        let val: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("invalid YAML for ConfigMap")?;
        let metadata = parse_metadata(&val)?;

        let data = parse_string_map(&val["data"]);

        let binary_data = parse_binary_data_map(&val["binaryData"]);

        Ok(ConfigMap {
            metadata,
            data,
            binary_data,
        })
    }

    /// Parse a multi-document YAML string (separated by `---`) into a vec of
    /// [`KubernetesResource`] values, dispatching each document by its `kind`.
    pub fn parse_multi_document(&self, yaml: &str) -> Result<Vec<KubernetesResource>> {
        let mut resources = Vec::new();

        for document in split_yaml_documents(yaml) {
            let trimmed = document.trim();
            if trimmed.is_empty() {
                continue;
            }

            let val: serde_yaml::Value =
                serde_yaml::from_str(trimmed).context("invalid YAML in multi-doc")?;

            let kind = val["kind"].as_str().unwrap_or("").to_string();

            let resource = match kind.as_str() {
                "Deployment" => {
                    let dep = self.parse_deployment(trimmed)?;
                    KubernetesResource::Deployment(dep)
                }
                "Service" => {
                    let svc = self.parse_service(trimmed)?;
                    KubernetesResource::Service(svc)
                }
                "Ingress" => {
                    let ing = self.parse_ingress(trimmed)?;
                    KubernetesResource::Ingress(ing)
                }
                "ConfigMap" => {
                    let cm = self.parse_config_map(trimmed)?;
                    KubernetesResource::ConfigMap(cm)
                }
                _ => KubernetesResource::Unknown(kind, val),
            };

            resources.push(resource);
        }

        Ok(resources)
    }

    /// Scan the environment variables and container images in a
    /// [`PodTemplateSpec`] for references to other services.
    ///
    /// Recognised patterns:
    /// - `<SERVICE>_SERVICE_HOST` / `<SERVICE>_SERVICE_PORT` env vars
    /// - Direct URLs matching `http(s)://<name>.<ns>.svc.cluster.local`
    pub fn extract_service_dependencies(&self, template: &PodTemplateSpec) -> Vec<ServiceId> {
        let mut deps: Vec<ServiceId> = Vec::new();
        let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();

        let host_re =
            Regex::new(r"^([A-Z][A-Z0-9_]*)_SERVICE_HOST$").expect("regex compile");
        let port_re =
            Regex::new(r"^([A-Z][A-Z0-9_]*)_SERVICE_PORT$").expect("regex compile");
        let url_re = Regex::new(
            r"https?://([a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)\.([a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)\.svc\.cluster\.local",
        )
        .expect("regex compile");

        let all_containers = template
            .containers
            .iter()
            .chain(template.init_containers.iter());

        for container in all_containers {
            for env in &container.env {
                // Check env var *name* for SERVICE_HOST / SERVICE_PORT pattern
                if let Some(caps) = host_re.captures(&env.name) {
                    let raw = caps.get(1).unwrap().as_str();
                    let svc_name = raw.to_lowercase().replace('_', "-");
                    let ns = template.metadata.namespace.clone();
                    let ns = if ns.is_empty() {
                        "default".to_string()
                    } else {
                        ns
                    };
                    if seen.insert((svc_name.clone(), ns.clone())) {
                        deps.push(ServiceId {
                            name: svc_name,
                            namespace: ns,
                        });
                    }
                }
                if let Some(caps) = port_re.captures(&env.name) {
                    let raw = caps.get(1).unwrap().as_str();
                    let svc_name = raw.to_lowercase().replace('_', "-");
                    let ns = template.metadata.namespace.clone();
                    let ns = if ns.is_empty() {
                        "default".to_string()
                    } else {
                        ns
                    };
                    if seen.insert((svc_name.clone(), ns.clone())) {
                        deps.push(ServiceId {
                            name: svc_name,
                            namespace: ns,
                        });
                    }
                }

                // Check env var *value* for cluster-local URL references
                if let Some(ref v) = env.value {
                    for caps in url_re.captures_iter(v) {
                        let svc_name = caps.get(1).unwrap().as_str().to_string();
                        let ns = caps.get(2).unwrap().as_str().to_string();
                        if seen.insert((svc_name.clone(), ns.clone())) {
                            deps.push(ServiceId {
                                name: svc_name,
                                namespace: ns,
                            });
                        }
                    }
                }
            }
        }

        deps
    }

    /// Extract a [`RetryPolicy`] from annotations that follow the
    /// `retry.cascade-verify/*` convention.
    pub fn extract_retry_config(
        &self,
        annotations: &IndexMap<String, String>,
    ) -> Option<RetryPolicy> {
        let prefix = "retry.cascade-verify/";

        let max_retries = annotations
            .get(&format!("{prefix}max-retries"))
            .and_then(|v| v.parse::<u32>().ok());
        let per_try = annotations
            .get(&format!("{prefix}per-try-timeout"))
            .and_then(|v| v.parse::<u64>().ok());
        let retry_on = annotations
            .get(&format!("{prefix}retry-on"))
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>());
        let backoff_base = annotations
            .get(&format!("{prefix}backoff-base"))
            .and_then(|v| v.parse::<u64>().ok());
        let backoff_max = annotations
            .get(&format!("{prefix}backoff-max"))
            .and_then(|v| v.parse::<u64>().ok());

        // Return None only when no relevant annotations exist at all.
        if max_retries.is_none()
            && per_try.is_none()
            && retry_on.is_none()
            && backoff_base.is_none()
            && backoff_max.is_none()
        {
            return None;
        }

        let defaults = RetryPolicy::default();
        Some(RetryPolicy {
            max_retries: max_retries.unwrap_or(defaults.max_retries),
            per_try_timeout_ms: per_try.unwrap_or(defaults.per_try_timeout_ms),
            retry_on: retry_on.unwrap_or(defaults.retry_on),
            backoff_base_ms: backoff_base.unwrap_or(defaults.backoff_base_ms),
            backoff_max_ms: backoff_max.unwrap_or(defaults.backoff_max_ms),
        })
    }

    /// Check whether a set of `labels` matches the given [`LabelSelector`].
    ///
    /// Both `match_labels` (exact equality) and `match_expressions` (In,
    /// NotIn, Exists, DoesNotExist) are evaluated.
    pub fn matches_selector(
        &self,
        selector: &LabelSelector,
        labels: &IndexMap<String, String>,
    ) -> bool {
        // All match_labels must be present with the expected value.
        for (key, expected) in &selector.match_labels {
            match labels.get(key) {
                Some(actual) if actual == expected => {}
                _ => return false,
            }
        }

        // All match_expressions must be satisfied.
        for req in &selector.match_expressions {
            let label_value = labels.get(&req.key);
            let satisfied = match req.operator.as_str() {
                "In" => match label_value {
                    Some(v) => req.values.contains(v),
                    None => false,
                },
                "NotIn" => match label_value {
                    Some(v) => !req.values.contains(v),
                    None => true,
                },
                "Exists" => label_value.is_some(),
                "DoesNotExist" => label_value.is_none(),
                _ => false,
            };
            if !satisfied {
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split a multi-document YAML string on `---` boundaries.
fn split_yaml_documents(yaml: &str) -> Vec<String> {
    let mut docs = Vec::new();
    let mut current = String::new();

    for line in yaml.lines() {
        if line.trim() == "---" {
            if !current.trim().is_empty() {
                docs.push(current.clone());
            }
            current.clear();
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }
    if !current.trim().is_empty() {
        docs.push(current);
    }
    docs
}

/// Extract [`ObjectMeta`] from the `metadata` key of a YAML value.
fn parse_metadata(val: &serde_yaml::Value) -> Result<ObjectMeta> {
    let meta = &val["metadata"];
    if meta.is_null() {
        bail!("resource is missing metadata");
    }

    let name = meta["name"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let namespace = meta["namespace"]
        .as_str()
        .unwrap_or("default")
        .to_string();

    let labels = parse_string_map(&meta["labels"]);
    let annotations = parse_string_map(&meta["annotations"]);

    let uid = meta["uid"].as_str().unwrap_or("").to_string();
    let resource_version = meta["resourceVersion"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(ObjectMeta {
        name,
        namespace,
        labels,
        annotations,
        uid,
        resource_version,
    })
}

/// Parse a flat string→string YAML mapping into an [`IndexMap`].
fn parse_string_map(val: &serde_yaml::Value) -> IndexMap<String, String> {
    let mut map = IndexMap::new();
    if let Some(mapping) = val.as_mapping() {
        for (k, v) in mapping {
            if let Some(key) = k.as_str() {
                let value = match v {
                    serde_yaml::Value::String(s) => s.clone(),
                    serde_yaml::Value::Number(n) => n.to_string(),
                    serde_yaml::Value::Bool(b) => b.to_string(),
                    _ => serde_yaml::to_string(v).unwrap_or_default(),
                };
                map.insert(key.to_string(), value);
            }
        }
    }
    map
}

/// Parse the `binaryData` section of a ConfigMap.
fn parse_binary_data_map(val: &serde_yaml::Value) -> IndexMap<String, Vec<u8>> {
    let mut map = IndexMap::new();
    if let Some(mapping) = val.as_mapping() {
        for (k, v) in mapping {
            if let Some(key) = k.as_str() {
                let bytes = match v.as_str() {
                    Some(s) => {
                        // binaryData values are base64-encoded in the K8s API.
                        // We store them decoded when possible, raw otherwise.
                        base64_decode(s).unwrap_or_else(|| s.as_bytes().to_vec())
                    }
                    None => Vec::new(),
                };
                map.insert(key.to_string(), bytes);
            }
        }
    }
    map
}

/// Minimal base64 decoder (no external crate required).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn val(c: u8) -> Option<u8> {
        TABLE.iter().position(|&b| b == c).map(|p| p as u8)
    }

    let clean: Vec<u8> = input.bytes().filter(|&b| b != b'=' && b != b'\n' && b != b'\r' && b != b' ').collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let chunks = clean.chunks(4);
    for chunk in chunks {
        let mut buf: u32 = 0;
        let mut count = 0u8;
        for &b in chunk {
            let v = val(b)?;
            buf = (buf << 6) | v as u32;
            count += 1;
        }
        // Shift remaining bits to the correct position.
        buf <<= (4 - count) * 6;
        let bytes_to_write = count.saturating_sub(1);
        for i in 0..bytes_to_write {
            out.push(((buf >> (16 - 8 * i)) & 0xFF) as u8);
        }
    }
    Some(out)
}

/// Parse a [`LabelSelector`] from a YAML value.
fn parse_label_selector(val: &serde_yaml::Value) -> LabelSelector {
    if val.is_null() {
        return LabelSelector::default();
    }

    let match_labels = parse_string_map(&val["matchLabels"]);

    let mut match_expressions = Vec::new();
    if let Some(seq) = val["matchExpressions"].as_sequence() {
        for item in seq {
            let key = item["key"].as_str().unwrap_or("").to_string();
            let operator = item["operator"].as_str().unwrap_or("").to_string();
            let values = item["values"]
                .as_sequence()
                .map(|vs| {
                    vs.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            match_expressions.push(LabelSelectorRequirement {
                key,
                operator,
                values,
            });
        }
    }

    LabelSelector {
        match_labels,
        match_expressions,
    }
}

/// Parse a [`PodTemplateSpec`] from its YAML value.
fn parse_pod_template(val: &serde_yaml::Value) -> Result<PodTemplateSpec> {
    let metadata = if val["metadata"].is_null() {
        ObjectMeta::default()
    } else {
        parse_metadata(val).unwrap_or_default()
    };

    let spec_val = &val["spec"];
    let containers = parse_containers(&spec_val["containers"])?;
    let init_containers = if spec_val["initContainers"].is_null() {
        Vec::new()
    } else {
        parse_containers(&spec_val["initContainers"])?
    };

    let volumes = parse_volumes(&spec_val["volumes"]);
    let service_account = spec_val["serviceAccountName"]
        .as_str()
        .or_else(|| spec_val["serviceAccount"].as_str())
        .map(String::from);

    Ok(PodTemplateSpec {
        metadata,
        containers,
        init_containers,
        volumes,
        service_account,
    })
}

/// Parse a [`DeploymentStrategy`] from its YAML value.
fn parse_strategy(val: &serde_yaml::Value) -> DeploymentStrategy {
    if val.is_null() {
        return DeploymentStrategy::default();
    }

    let strategy_type = val["type"].as_str().unwrap_or("RollingUpdate").to_string();

    let (max_unavailable, max_surge) = if strategy_type == "RollingUpdate" {
        let ru = &val["rollingUpdate"];
        let mu = yaml_value_to_string(&ru["maxUnavailable"]);
        let ms = yaml_value_to_string(&ru["maxSurge"]);
        (mu, ms)
    } else {
        (None, None)
    };

    DeploymentStrategy {
        strategy_type,
        max_unavailable,
        max_surge,
    }
}

/// Parse an array of container specs from a YAML sequence value.
fn parse_containers(val: &serde_yaml::Value) -> Result<Vec<ContainerSpec>> {
    let mut containers = Vec::new();
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Ok(containers),
    };

    for item in seq {
        let name = item["name"].as_str().unwrap_or("").to_string();
        let image = item["image"].as_str().unwrap_or("").to_string();

        let ports = parse_container_ports(&item["ports"]);
        let resources = parse_resources(&item["resources"]);
        let env = parse_env_vars(&item["env"]);
        let probes = parse_probes(item);

        let command = parse_string_seq(&item["command"]);
        let args = parse_string_seq(&item["args"]);

        let volume_mounts = parse_volume_mounts(&item["volumeMounts"]);

        containers.push(ContainerSpec {
            name,
            image,
            ports,
            resources,
            env,
            probes,
            command,
            args,
            volume_mounts,
        });
    }

    Ok(containers)
}

/// Parse a YAML sequence of strings (e.g. `command`, `args`).
fn parse_string_seq(val: &serde_yaml::Value) -> Vec<String> {
    val.as_sequence()
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse container port definitions.
fn parse_container_ports(val: &serde_yaml::Value) -> Vec<ContainerPort> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| ContainerPort {
            name: item["name"].as_str().map(String::from),
            container_port: item["containerPort"].as_u64().unwrap_or(0) as u16,
            protocol: item["protocol"].as_str().unwrap_or("TCP").to_string(),
        })
        .collect()
}

/// Parse CPU / memory resource requirements.
fn parse_resources(val: &serde_yaml::Value) -> ResourceRequirements {
    if val.is_null() {
        return ResourceRequirements::default();
    }

    let requests = &val["requests"];
    let limits = &val["limits"];

    ResourceRequirements {
        cpu_request: yaml_value_to_string(&requests["cpu"]),
        memory_request: yaml_value_to_string(&requests["memory"]),
        cpu_limit: yaml_value_to_string(&limits["cpu"]),
        memory_limit: yaml_value_to_string(&limits["memory"]),
    }
}

/// Parse liveness, readiness, and startup probes from a container value.
fn parse_probes(container_val: &serde_yaml::Value) -> ContainerProbes {
    ContainerProbes {
        liveness: parse_single_probe(&container_val["livenessProbe"]),
        readiness: parse_single_probe(&container_val["readinessProbe"]),
        startup: parse_single_probe(&container_val["startupProbe"]),
    }
}

/// Parse a single probe definition.
fn parse_single_probe(val: &serde_yaml::Value) -> Option<Probe> {
    if val.is_null() {
        return None;
    }

    let http_get = if !val["httpGet"].is_null() {
        Some(HttpGetAction {
            path: val["httpGet"]["path"]
                .as_str()
                .unwrap_or("/")
                .to_string(),
            port: val["httpGet"]["port"].as_u64().unwrap_or(80) as u16,
        })
    } else {
        None
    };

    let tcp_socket = if !val["tcpSocket"].is_null() {
        Some(TcpSocketAction {
            port: val["tcpSocket"]["port"].as_u64().unwrap_or(0) as u16,
        })
    } else {
        None
    };

    Some(Probe {
        http_get,
        tcp_socket,
        initial_delay_seconds: val["initialDelaySeconds"].as_u64().unwrap_or(0) as u32,
        period_seconds: val["periodSeconds"].as_u64().unwrap_or(10) as u32,
        timeout_seconds: val["timeoutSeconds"].as_u64().unwrap_or(1) as u32,
        failure_threshold: val["failureThreshold"].as_u64().unwrap_or(3) as u32,
    })
}

/// Parse environment variable definitions.
fn parse_env_vars(val: &serde_yaml::Value) -> Vec<EnvVar> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| {
            let name = item["name"].as_str().unwrap_or("").to_string();
            let value = item["value"].as_str().map(String::from);

            let value_from = if !item["valueFrom"].is_null() {
                let vf = &item["valueFrom"];
                let config_map_key_ref = if !vf["configMapKeyRef"].is_null() {
                    Some(KeyRef {
                        name: vf["configMapKeyRef"]["name"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        key: vf["configMapKeyRef"]["key"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                    })
                } else {
                    None
                };

                let secret_key_ref = if !vf["secretKeyRef"].is_null() {
                    Some(KeyRef {
                        name: vf["secretKeyRef"]["name"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        key: vf["secretKeyRef"]["key"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                    })
                } else {
                    None
                };

                let field_ref = if !vf["fieldRef"].is_null() {
                    Some(FieldRef {
                        field_path: vf["fieldRef"]["fieldPath"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                    })
                } else {
                    None
                };

                Some(EnvVarSource {
                    config_map_key_ref,
                    secret_key_ref,
                    field_ref,
                })
            } else {
                None
            };

            EnvVar {
                name,
                value,
                value_from,
            }
        })
        .collect()
}

/// Parse volume definitions from the pod spec.
fn parse_volumes(val: &serde_yaml::Value) -> Vec<Volume> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .filter_map(|item| {
            let name = item["name"].as_str()?.to_string();

            let volume_type = if !item["configMap"].is_null() {
                let cm_name = item["configMap"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                VolumeType::ConfigMap(cm_name)
            } else if !item["secret"].is_null() {
                let secret_name = item["secret"]["secretName"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                VolumeType::Secret(secret_name)
            } else if !item["emptyDir"].is_null() {
                VolumeType::EmptyDir
            } else if !item["persistentVolumeClaim"].is_null() {
                let claim = item["persistentVolumeClaim"]["claimName"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                VolumeType::PersistentVolumeClaim(claim)
            } else {
                VolumeType::EmptyDir
            };

            Some(Volume { name, volume_type })
        })
        .collect()
}

/// Parse volume mount entries.
fn parse_volume_mounts(val: &serde_yaml::Value) -> Vec<VolumeMount> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .filter_map(|item| {
            let name = item["name"].as_str()?.to_string();
            let mount_path = item["mountPath"].as_str().unwrap_or("").to_string();
            let read_only = item["readOnly"].as_bool().unwrap_or(false);
            Some(VolumeMount {
                name,
                mount_path,
                read_only,
            })
        })
        .collect()
}

/// Parse Service port definitions.
fn parse_service_ports(val: &serde_yaml::Value) -> Vec<ServicePort> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| {
            let port = item["port"].as_u64().unwrap_or(0) as u16;
            let target_port = item["targetPort"].as_u64().unwrap_or(port as u64) as u16;
            ServicePort {
                name: item["name"].as_str().map(String::from),
                port,
                target_port,
                protocol: item["protocol"].as_str().unwrap_or("TCP").to_string(),
                node_port: item["nodePort"].as_u64().map(|v| v as u16),
            }
        })
        .collect()
}

/// Parse Ingress rules from the spec.
fn parse_ingress_rules(val: &serde_yaml::Value) -> Vec<IngressRule> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| {
            let host = item["host"].as_str().map(String::from);
            let paths = parse_ingress_paths(&item["http"]["paths"]);
            IngressRule { host, paths }
        })
        .collect()
}

/// Parse Ingress path entries inside a rule.
fn parse_ingress_paths(val: &serde_yaml::Value) -> Vec<IngressPath> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| {
            let path = item["path"].as_str().unwrap_or("/").to_string();
            let path_type = item["pathType"].as_str().unwrap_or("Prefix").to_string();
            let backend_val = &item["backend"];

            // Support both networking.k8s.io/v1 and extensions/v1beta1 style.
            let (service_name, service_port) =
                if !backend_val["service"].is_null() {
                    let sn = backend_val["service"]["name"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let sp = backend_val["service"]["port"]["number"]
                        .as_u64()
                        .unwrap_or(80) as u16;
                    (sn, sp)
                } else {
                    let sn = backend_val["serviceName"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let sp = backend_val["servicePort"]
                        .as_u64()
                        .unwrap_or(80) as u16;
                    (sn, sp)
                };

            IngressPath {
                path,
                path_type,
                backend: IngressBackend {
                    service_name,
                    service_port,
                },
            }
        })
        .collect()
}

/// Parse Ingress TLS section.
fn parse_ingress_tls(val: &serde_yaml::Value) -> Vec<IngressTLS> {
    let seq = match val.as_sequence() {
        Some(s) => s,
        None => return Vec::new(),
    };

    seq.iter()
        .map(|item| {
            let hosts = item["hosts"]
                .as_sequence()
                .map(|h| {
                    h.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let secret_name = item["secretName"]
                .as_str()
                .unwrap_or("")
                .to_string();
            IngressTLS { hosts, secret_name }
        })
        .collect()
}

/// Convert a [`serde_yaml::Value`] that may be a string or number into an
/// `Option<String>`.
fn yaml_value_to_string(val: &serde_yaml::Value) -> Option<String> {
    match val {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> KubernetesParser {
        KubernetesParser::new()
    }

    // -- Deployment ----------------------------------------------------------

    #[test]
    fn test_parse_deployment_basic() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
  labels:
    app: my-app
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: web
        image: nginx:1.25
        ports:
        - containerPort: 80
          name: http
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        assert_eq!(dep.metadata.name, "my-app");
        assert_eq!(dep.metadata.namespace, "production");
        assert_eq!(dep.spec.replicas, 3);
        assert_eq!(dep.spec.selector.match_labels.get("app").unwrap(), "my-app");
        assert_eq!(dep.spec.template.containers.len(), 1);
        assert_eq!(dep.spec.template.containers[0].name, "web");
        assert_eq!(dep.spec.template.containers[0].image, "nginx:1.25");
        assert_eq!(dep.spec.template.containers[0].ports[0].container_port, 80);
        assert_eq!(
            dep.spec.template.containers[0].ports[0].name.as_deref(),
            Some("http")
        );
    }

    #[test]
    fn test_parse_deployment_with_resources() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resource-app
  namespace: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: resource-app
  template:
    metadata:
      labels:
        app: resource-app
    spec:
      containers:
      - name: api
        image: api:latest
        resources:
          requests:
            cpu: "250m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "256Mi"
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        let res = &dep.spec.template.containers[0].resources;
        assert_eq!(res.cpu_request.as_deref(), Some("250m"));
        assert_eq!(res.memory_request.as_deref(), Some("128Mi"));
        assert_eq!(res.cpu_limit.as_deref(), Some("500m"));
        assert_eq!(res.memory_limit.as_deref(), Some("256Mi"));
    }

    #[test]
    fn test_parse_deployment_with_probes() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: probe-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: probe-app
  template:
    metadata:
      labels:
        app: probe-app
    spec:
      containers:
      - name: server
        image: server:v2
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
          timeoutSeconds: 3
          failureThreshold: 5
        readinessProbe:
          tcpSocket:
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        let probes = &dep.spec.template.containers[0].probes;

        let liveness = probes.liveness.as_ref().unwrap();
        assert_eq!(liveness.http_get.as_ref().unwrap().path, "/healthz");
        assert_eq!(liveness.http_get.as_ref().unwrap().port, 8080);
        assert_eq!(liveness.initial_delay_seconds, 15);
        assert_eq!(liveness.period_seconds, 20);
        assert_eq!(liveness.timeout_seconds, 3);
        assert_eq!(liveness.failure_threshold, 5);

        let readiness = probes.readiness.as_ref().unwrap();
        assert!(readiness.http_get.is_none());
        assert_eq!(readiness.tcp_socket.as_ref().unwrap().port, 8080);
        assert_eq!(readiness.initial_delay_seconds, 5);
        assert_eq!(readiness.period_seconds, 10);

        assert!(probes.startup.is_none());
    }

    #[test]
    fn test_parse_deployment_with_env_vars() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: env-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: env-app
  template:
    metadata:
      labels:
        app: env-app
    spec:
      containers:
      - name: worker
        image: worker:latest
        env:
        - name: DATABASE_URL
          value: "postgres://db:5432/mydb"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: my-secret
              key: api-key
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: log-level
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        let envs = &dep.spec.template.containers[0].env;
        assert_eq!(envs.len(), 4);

        assert_eq!(envs[0].name, "DATABASE_URL");
        assert_eq!(envs[0].value.as_deref(), Some("postgres://db:5432/mydb"));
        assert!(envs[0].value_from.is_none());

        assert_eq!(envs[1].name, "API_KEY");
        let vf1 = envs[1].value_from.as_ref().unwrap();
        assert_eq!(vf1.secret_key_ref.as_ref().unwrap().name, "my-secret");
        assert_eq!(vf1.secret_key_ref.as_ref().unwrap().key, "api-key");

        assert_eq!(envs[2].name, "LOG_LEVEL");
        let vf2 = envs[2].value_from.as_ref().unwrap();
        assert_eq!(vf2.config_map_key_ref.as_ref().unwrap().name, "app-config");
        assert_eq!(vf2.config_map_key_ref.as_ref().unwrap().key, "log-level");

        assert_eq!(envs[3].name, "POD_NAME");
        let vf3 = envs[3].value_from.as_ref().unwrap();
        assert_eq!(vf3.field_ref.as_ref().unwrap().field_path, "metadata.name");
    }

    // -- Service -------------------------------------------------------------

    #[test]
    fn test_parse_service_clusterip() {
        let yaml = r#"
apiVersion: v1
kind: Service
metadata:
  name: my-svc
  namespace: default
spec:
  type: ClusterIP
  selector:
    app: my-app
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  clusterIP: 10.96.0.42
"#;
        let svc = parser().parse_service(yaml).unwrap();
        assert_eq!(svc.metadata.name, "my-svc");
        assert_eq!(svc.spec.service_type, "ClusterIP");
        assert_eq!(svc.spec.selector.get("app").unwrap(), "my-app");
        assert_eq!(svc.spec.ports.len(), 1);
        assert_eq!(svc.spec.ports[0].port, 80);
        assert_eq!(svc.spec.ports[0].target_port, 8080);
        assert_eq!(svc.spec.ports[0].protocol, "TCP");
        assert_eq!(svc.spec.cluster_ip.as_deref(), Some("10.96.0.42"));
    }

    #[test]
    fn test_parse_service_nodeport() {
        let yaml = r#"
apiVersion: v1
kind: Service
metadata:
  name: external-svc
  namespace: default
spec:
  type: NodePort
  selector:
    app: external-app
  ports:
  - name: http
    port: 80
    targetPort: 8080
    nodePort: 30080
  - name: https
    port: 443
    targetPort: 8443
    nodePort: 30443
"#;
        let svc = parser().parse_service(yaml).unwrap();
        assert_eq!(svc.spec.service_type, "NodePort");
        assert_eq!(svc.spec.ports.len(), 2);
        assert_eq!(svc.spec.ports[0].node_port, Some(30080));
        assert_eq!(svc.spec.ports[1].port, 443);
        assert_eq!(svc.spec.ports[1].node_port, Some(30443));
    }

    // -- Ingress -------------------------------------------------------------

    #[test]
    fn test_parse_ingress() {
        let yaml = r#"
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  tls:
  - hosts:
    - example.com
    - www.example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-svc
            port:
              number: 80
      - path: /web
        pathType: Exact
        backend:
          service:
            name: web-svc
            port:
              number: 8080
"#;
        let ing = parser().parse_ingress(yaml).unwrap();
        assert_eq!(ing.metadata.name, "my-ingress");
        assert_eq!(ing.tls.len(), 1);
        assert_eq!(ing.tls[0].hosts, vec!["example.com", "www.example.com"]);
        assert_eq!(ing.tls[0].secret_name, "tls-secret");
        assert_eq!(ing.rules.len(), 1);
        assert_eq!(ing.rules[0].host.as_deref(), Some("example.com"));
        assert_eq!(ing.rules[0].paths.len(), 2);
        assert_eq!(ing.rules[0].paths[0].path, "/api");
        assert_eq!(ing.rules[0].paths[0].path_type, "Prefix");
        assert_eq!(ing.rules[0].paths[0].backend.service_name, "api-svc");
        assert_eq!(ing.rules[0].paths[0].backend.service_port, 80);
        assert_eq!(ing.rules[0].paths[1].path, "/web");
        assert_eq!(ing.rules[0].paths[1].path_type, "Exact");
        assert_eq!(ing.rules[0].paths[1].backend.service_name, "web-svc");
    }

    // -- ConfigMap -----------------------------------------------------------

    #[test]
    fn test_parse_config_map() {
        let yaml = r#"
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  log-level: info
  max-connections: "100"
  config.yaml: |
    server:
      port: 8080
"#;
        let cm = parser().parse_config_map(yaml).unwrap();
        assert_eq!(cm.metadata.name, "app-config");
        assert_eq!(cm.metadata.namespace, "production");
        assert_eq!(cm.data.get("log-level").unwrap(), "info");
        assert_eq!(cm.data.get("max-connections").unwrap(), "100");
        assert!(cm.data.get("config.yaml").unwrap().contains("port: 8080"));
    }

    // -- Multi-document ------------------------------------------------------

    #[test]
    fn test_parse_multi_document() {
        let yaml = r#"---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cfg
  namespace: default
data:
  key: value
---
apiVersion: v1
kind: Service
metadata:
  name: svc
  namespace: default
spec:
  selector:
    app: test
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dep
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: main
        image: test:latest
"#;
        let resources = parser().parse_multi_document(yaml).unwrap();
        assert_eq!(resources.len(), 3);

        assert!(matches!(&resources[0], KubernetesResource::ConfigMap(cm) if cm.metadata.name == "cfg"));
        assert!(matches!(&resources[1], KubernetesResource::Service(svc) if svc.metadata.name == "svc"));
        assert!(matches!(&resources[2], KubernetesResource::Deployment(dep) if dep.metadata.name == "dep"));
    }

    // -- Service dependencies ------------------------------------------------

    #[test]
    fn test_extract_service_dependencies() {
        let template = PodTemplateSpec {
            metadata: ObjectMeta {
                namespace: "production".to_string(),
                ..Default::default()
            },
            containers: vec![ContainerSpec {
                name: "app".to_string(),
                image: "app:latest".to_string(),
                ports: vec![],
                resources: ResourceRequirements::default(),
                env: vec![
                    EnvVar {
                        name: "REDIS_SERVICE_HOST".to_string(),
                        value: Some("10.0.0.1".to_string()),
                        value_from: None,
                    },
                    EnvVar {
                        name: "REDIS_SERVICE_PORT".to_string(),
                        value: Some("6379".to_string()),
                        value_from: None,
                    },
                    EnvVar {
                        name: "AUTH_API_URL".to_string(),
                        value: Some(
                            "http://auth-api.auth-ns.svc.cluster.local:8080/v1".to_string(),
                        ),
                        value_from: None,
                    },
                    EnvVar {
                        name: "PLAIN_VAR".to_string(),
                        value: Some("hello".to_string()),
                        value_from: None,
                    },
                ],
                probes: ContainerProbes::default(),
                command: vec![],
                args: vec![],
                volume_mounts: vec![],
            }],
            init_containers: vec![],
            volumes: vec![],
            service_account: None,
        };

        let deps = parser().extract_service_dependencies(&template);
        assert_eq!(deps.len(), 2);

        let redis = deps.iter().find(|d| d.name == "redis").unwrap();
        assert_eq!(redis.namespace, "production");

        let auth = deps.iter().find(|d| d.name == "auth-api").unwrap();
        assert_eq!(auth.namespace, "auth-ns");
    }

    // -- Retry config --------------------------------------------------------

    #[test]
    fn test_extract_retry_config() {
        let mut annotations = IndexMap::new();
        annotations.insert(
            "retry.cascade-verify/max-retries".to_string(),
            "5".to_string(),
        );
        annotations.insert(
            "retry.cascade-verify/per-try-timeout".to_string(),
            "2000".to_string(),
        );
        annotations.insert(
            "retry.cascade-verify/retry-on".to_string(),
            "5xx,reset,connect-failure".to_string(),
        );
        annotations.insert(
            "retry.cascade-verify/backoff-base".to_string(),
            "50".to_string(),
        );
        annotations.insert(
            "retry.cascade-verify/backoff-max".to_string(),
            "500".to_string(),
        );

        let policy = parser().extract_retry_config(&annotations).unwrap();
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.per_try_timeout_ms, 2000);
        assert_eq!(
            policy.retry_on,
            vec!["5xx", "reset", "connect-failure"]
        );
        assert_eq!(policy.backoff_base_ms, 50);
        assert_eq!(policy.backoff_max_ms, 500);

        // Empty annotations should return None.
        let empty: IndexMap<String, String> = IndexMap::new();
        assert!(parser().extract_retry_config(&empty).is_none());
    }

    // -- Selector matching ---------------------------------------------------

    #[test]
    fn test_matches_selector_basic() {
        let selector = LabelSelector {
            match_labels: {
                let mut m = IndexMap::new();
                m.insert("app".to_string(), "web".to_string());
                m.insert("env".to_string(), "prod".to_string());
                m
            },
            match_expressions: vec![],
        };

        let mut matching = IndexMap::new();
        matching.insert("app".to_string(), "web".to_string());
        matching.insert("env".to_string(), "prod".to_string());
        matching.insert("extra".to_string(), "ok".to_string());
        assert!(parser().matches_selector(&selector, &matching));

        let mut non_matching = IndexMap::new();
        non_matching.insert("app".to_string(), "web".to_string());
        non_matching.insert("env".to_string(), "staging".to_string());
        assert!(!parser().matches_selector(&selector, &non_matching));

        let mut missing_key = IndexMap::new();
        missing_key.insert("app".to_string(), "web".to_string());
        assert!(!parser().matches_selector(&selector, &missing_key));
    }

    #[test]
    fn test_matches_selector_expressions() {
        let selector = LabelSelector {
            match_labels: IndexMap::new(),
            match_expressions: vec![
                LabelSelectorRequirement {
                    key: "tier".to_string(),
                    operator: "In".to_string(),
                    values: vec!["frontend".to_string(), "backend".to_string()],
                },
                LabelSelectorRequirement {
                    key: "deprecated".to_string(),
                    operator: "DoesNotExist".to_string(),
                    values: vec![],
                },
                LabelSelectorRequirement {
                    key: "release".to_string(),
                    operator: "Exists".to_string(),
                    values: vec![],
                },
                LabelSelectorRequirement {
                    key: "version".to_string(),
                    operator: "NotIn".to_string(),
                    values: vec!["alpha".to_string(), "beta".to_string()],
                },
            ],
        };

        let mut good = IndexMap::new();
        good.insert("tier".to_string(), "frontend".to_string());
        good.insert("release".to_string(), "stable".to_string());
        good.insert("version".to_string(), "v1".to_string());
        assert!(parser().matches_selector(&selector, &good));

        // Fail: tier not in allowed set
        let mut bad_tier = IndexMap::new();
        bad_tier.insert("tier".to_string(), "middleware".to_string());
        bad_tier.insert("release".to_string(), "stable".to_string());
        bad_tier.insert("version".to_string(), "v1".to_string());
        assert!(!parser().matches_selector(&selector, &bad_tier));

        // Fail: deprecated key exists
        let mut bad_deprecated = IndexMap::new();
        bad_deprecated.insert("tier".to_string(), "backend".to_string());
        bad_deprecated.insert("release".to_string(), "stable".to_string());
        bad_deprecated.insert("version".to_string(), "v1".to_string());
        bad_deprecated.insert("deprecated".to_string(), "true".to_string());
        assert!(!parser().matches_selector(&selector, &bad_deprecated));

        // Fail: release label missing (Exists requirement)
        let mut bad_release = IndexMap::new();
        bad_release.insert("tier".to_string(), "frontend".to_string());
        bad_release.insert("version".to_string(), "v1".to_string());
        assert!(!parser().matches_selector(&selector, &bad_release));

        // Fail: version is in the disallowed set
        let mut bad_version = IndexMap::new();
        bad_version.insert("tier".to_string(), "backend".to_string());
        bad_version.insert("release".to_string(), "stable".to_string());
        bad_version.insert("version".to_string(), "alpha".to_string());
        assert!(!parser().matches_selector(&selector, &bad_version));
    }

    // -- Strategy ------------------------------------------------------------

    #[test]
    fn test_parse_deployment_strategy() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategy-app
  namespace: default
spec:
  replicas: 4
  selector:
    matchLabels:
      app: strategy-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: "1"
      maxSurge: "2"
  template:
    metadata:
      labels:
        app: strategy-app
    spec:
      containers:
      - name: main
        image: app:latest
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        assert_eq!(dep.spec.strategy.strategy_type, "RollingUpdate");
        assert_eq!(dep.spec.strategy.max_unavailable.as_deref(), Some("1"));
        assert_eq!(dep.spec.strategy.max_surge.as_deref(), Some("2"));

        // Recreate strategy
        let yaml_recreate = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recreate-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: recreate-app
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: recreate-app
    spec:
      containers:
      - name: main
        image: app:latest
"#;
        let dep2 = parser().parse_deployment(yaml_recreate).unwrap();
        assert_eq!(dep2.spec.strategy.strategy_type, "Recreate");
        assert!(dep2.spec.strategy.max_unavailable.is_none());
        assert!(dep2.spec.strategy.max_surge.is_none());
    }

    // -- Init containers -----------------------------------------------------

    #[test]
    fn test_parse_init_containers() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: init-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: init-app
  template:
    metadata:
      labels:
        app: init-app
    spec:
      serviceAccountName: my-sa
      initContainers:
      - name: db-migrate
        image: migrate:latest
        command: ["./migrate"]
        args: ["--direction", "up"]
        env:
        - name: DB_HOST
          value: "postgres"
        volumeMounts:
        - name: config-vol
          mountPath: /etc/config
          readOnly: true
      containers:
      - name: server
        image: server:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: config-vol
          mountPath: /etc/config
          readOnly: true
      volumes:
      - name: config-vol
        configMap:
          name: server-config
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        let tmpl = &dep.spec.template;

        assert_eq!(tmpl.service_account.as_deref(), Some("my-sa"));

        assert_eq!(tmpl.init_containers.len(), 1);
        let init = &tmpl.init_containers[0];
        assert_eq!(init.name, "db-migrate");
        assert_eq!(init.image, "migrate:latest");
        assert_eq!(init.command, vec!["./migrate"]);
        assert_eq!(init.args, vec!["--direction", "up"]);
        assert_eq!(init.env.len(), 1);
        assert_eq!(init.env[0].name, "DB_HOST");
        assert_eq!(init.volume_mounts.len(), 1);
        assert_eq!(init.volume_mounts[0].name, "config-vol");
        assert!(init.volume_mounts[0].read_only);

        assert_eq!(tmpl.containers.len(), 1);
        assert_eq!(tmpl.containers[0].name, "server");

        assert_eq!(tmpl.volumes.len(), 1);
        assert_eq!(tmpl.volumes[0].name, "config-vol");
        assert_eq!(
            tmpl.volumes[0].volume_type,
            VolumeType::ConfigMap("server-config".to_string())
        );
    }

    // -- Additional coverage -------------------------------------------------

    #[test]
    fn test_parse_deployment_missing_spec_errors() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bad
"#;
        assert!(parser().parse_deployment(yaml).is_err());
    }

    #[test]
    fn test_matches_selector_empty_selector() {
        let selector = LabelSelector::default();
        let labels = IndexMap::new();
        // An empty selector matches everything.
        assert!(parser().matches_selector(&selector, &labels));
    }

    #[test]
    fn test_parse_multi_document_unknown_kind() {
        let yaml = r#"---
apiVersion: v1
kind: Namespace
metadata:
  name: test-ns
"#;
        let resources = parser().parse_multi_document(yaml).unwrap();
        assert_eq!(resources.len(), 1);
        assert!(matches!(&resources[0], KubernetesResource::Unknown(kind, _) if kind == "Namespace"));
    }

    #[test]
    fn test_parse_volumes_all_types() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vol-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vol-app
  template:
    metadata:
      labels:
        app: vol-app
    spec:
      containers:
      - name: main
        image: app:latest
      volumes:
      - name: cfg
        configMap:
          name: my-cm
      - name: sec
        secret:
          secretName: my-secret
      - name: tmp
        emptyDir: {}
      - name: data
        persistentVolumeClaim:
          claimName: my-pvc
"#;
        let dep = parser().parse_deployment(yaml).unwrap();
        let vols = &dep.spec.template.volumes;
        assert_eq!(vols.len(), 4);
        assert_eq!(vols[0].volume_type, VolumeType::ConfigMap("my-cm".to_string()));
        assert_eq!(vols[1].volume_type, VolumeType::Secret("my-secret".to_string()));
        assert_eq!(vols[2].volume_type, VolumeType::EmptyDir);
        assert_eq!(
            vols[3].volume_type,
            VolumeType::PersistentVolumeClaim("my-pvc".to_string())
        );
    }
}
